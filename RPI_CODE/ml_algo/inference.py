import threading
import time
import struct
import os 
import subprocess
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import resampy
import numpy as np
import params
import yamnet as yamnet_model
import copy

from scipy.io.wavfile import write

path_append = 'ml_algo/'

running_as_main = False

def PRINT(str):
    if(running_as_main):
        print(str)

'''def singleton(cls, *args, **kw):
    instances = {}
    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton

@singleton'''
class AudioProcessor:
    def audio_predict(self):
        print("Starting prediction thread")
        self.ynet = yamnet_model.yamnet_frames_model(params)
        self.ynet.load_weights(path_append + 'yamnet.h5')
        self.yamnet_classes = yamnet_model.class_names(path_append + 'yamnet_class_map.csv')
        self.proc_event.set()
        print("prediction thread ready")
        while(True):
            self.proc_event.wait()
            self.proc_event.clear()
            self.proc_ready = False
            PRINT("PROCESSING")
            time_now = time.time()
            
            recording = np.asarray(self.proc_data, dtype = np.int16)
            write('output.wav', self.sample_rate, recording)
            recording = recording.reshape((recording.shape[0], 1))
            waveform = recording / 32768.0
            
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            if self.sample_rate != params.SAMPLE_RATE:
                waveform = resampy.resample(waveform, self.sample_rate, params.SAMPLE_RATE)

            scores, _ = self.ynet.predict(np.reshape(waveform, [1, -1]), steps=1)
            prediction = np.mean(scores, axis=0)
            
            top5_i = np.argsort(prediction)[::-1][:5]
            top5_classes = [self.yamnet_classes[i] for i in top5_i] 
            if("Gunshot, gunfire" in top5_classes):
                likely_fireworks = False
                for top_class in top5_classes:
                    if (top_class == "Fireworks" or top_class == "Firecracker"):
                        likely_fireworks = True
                    if(top_class == "Gunshot, gunfire"):
                        break
                if likely_fireworks == False:
                    print("***************GUNSHOT PREDICTED*************")
                    prediction = prediction - prediction.mean(axis=0)
                    prediction = prediction / np.abs(prediction).max(axis=0)
                    self.prediction_value = (prediction[421], np.max(recording))
                    print("Confidence coefficient: {}\nMax volume: {}".format(self.prediction_value[0], self.prediction_value[1]));
                    self.gunshot_event.set()
                else:
                    print("************MOST LIKELY FIREWORKS************")
            elif("Fireworks" in top5_classes or "Firecrackers" in top5_classes):
                print("*************FIREWORKS PREDICTED************")
            #print("Prediction:", ":\n" + '\n'.join(' {:12s}: {:.3f}'.format(self.yamnet_classes[i], prediction[i]) for i in top5_i))
            PRINT("Prediction took {} seconds".format(time.time() - time_now))

            PRINT("DONE")
            self.proc_ready = True

    def audio_process(self):
        if not os.path.exists("audio.fifo"):
            print("CREATING FIFO")
            os.mkfifo("audio.fifo")
        
        threshold_crossed = False
        
        counter = 0
        counter_max_th = self.sample_rate * self.duration
        counter_transmit_th = counter_max_th * 3 / 4

        self.proc_event.wait()

        print("Starting arecord at {} HZ".format(self.sample_rate))
        proc = subprocess.Popen('arecord -D plug:default -f S16_LE -c 1 -r {} -t raw audio.fifo'.format(self.sample_rate).split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("Reading from mic")
        with open("audio.fifo", "rb") as fifo:
            while(True):
                line = fifo.read(2)
                if(len(line) == 2):
                    val = struct.unpack("<h", line)[0]
                    if(abs(val) > self.threshold):
                        threshold_crossed = True
                        if(counter == 0):
                            PRINT("Threshold crossed, capturing more audio")
                            self.gunshot_time = time.time()
                        if(counter > counter_transmit_th):
                            counter = counter_transmit_th
                            self.gunshot_time = time.time()
                    if(threshold_crossed):
                        counter += 1
                        if(self.proc_ready):
                            if(counter > counter_transmit_th):
                                PRINT("Triggering ML Algorithm")
                                self.proc_data = copy.deepcopy(list(self.data))
                                self.proc_event.set()
                                counter = 0
                                threshold_crossed = False
                        else:
                            #if(abs(val) > self.threshold):
                                #print("Threshold crossed while processing")
                            if(counter > counter_max_th):
                                print("WARNING: Threshold crossing missed")
                                counter = 0
                                threshold_crossed = False
                    self.data.append(val)

    def __init__(self, th=2000, sr=16000, dur=1):
        self.sample_rate = sr
        self.threshold = th
        self.duration = dur
        self.data = collections.deque(maxlen = dur*sr)
        self.proc_data = [0]*dur*sr
        self.gunshot_time = None 
        self.gunshot_event = threading.Event()
        self.proc_event = threading.Event()
        self.proc_ready = True
        self.prediction_value = ()

        self.audio_process_thread = threading.Thread(target=self.audio_process)
        self.audio_process_thread.daemon = True
        self.audio_predict_thread = threading.Thread(target=self.audio_predict)
        self.audio_predict_thread.daemo = True
        
        self.audio_process_thread.start() 
        self.audio_predict_thread.start()
        

if __name__ == "__main__":
    running_as_main = True
    path_append = ''
    ap = AudioProcessor(th=17000, sr=16000)
    while(True):
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            ap.proc_event.set()
            break


    
