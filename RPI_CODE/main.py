
import sys

sys.path.append('comm')
sys.path.append('ml_algo')
#import pi_to_dot as ptd
import inference as infer
import time
import threading
import serial
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

device = '/dev/ttyACM1'
ser = serial.Serial(device)

def init_connection():
    ser.baudrate = 9600
    ser.flushOutput()
    ser.flushInput()
    pass

# pass in confidence and max_val as numbers and they can be 
# converted to string
def get_formatted_data(confidence):
    text = "C:" + str(confidence)
    return text

# takes in a string and send to the dot
def send_to_dot(to_send):
    line = "not"
    # holds the pi here until the dot is able to transmit
    while "ready" not in line:
        line = ser.readline()
        line = line.decode('utf-8', errors='ignore')
    print("sending: ")
    to_send += "!"
    print(to_send)
    ser.write(to_send.encode())

def get_formatted_time(t, sync=False):
    if(sync):
        to_send = "S:" + str(t)
    else:
        to_send = "T:" + str(t)
    return to_send

def time_sync():
    while(True):
        print("Sending current time for synchronization")
        #send_to_dot(get_formatted_time(time.time(), True))
        time.sleep(60)

def main():
    ap = infer.AudioProcessor(th=1000, sr=16000)
    init_connection()
    sync_thread = threading.Thread(target = time_sync)
    sync_thread.daemon = True
    sync_thread.start()
    while(True):
        ap.gunshot_event.wait()
        ap.gunshot_event.clear()
        print("Gunshot event detected! Sending to xdot")
        formatted_value = get_formatted_data(ap.prediction_value[0])
        formatted_time = get_formatted_time(ap.gunshot_time)
        send_to_dot(formatted_value + "," + formatted_time)
        print("Data is sent")

if __name__ == "__main__":
    main()
