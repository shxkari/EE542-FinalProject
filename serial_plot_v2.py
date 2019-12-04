import serial 
from serial import Serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

device_name = '/dev/ttyXRUSB0'
ser = serial.Serial(device_name)

ser.baudrate = 115200
ser.close()
ser.open()
style.use('fivethirtyeight')
#plt.ylim(0,32000)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_ylim([0,32000])
def animate(i):
	#graph_data = open('example.txt','r').read()
	#lines = graph_data.split('\n')
	xs = []
	ys = []
	count= 0
	for count in range (20):
		#print("reached while\n")
		# Read a line and strip whitespace from it
		readin = ser.readline()
		if readin:  # If it isn't a blank line
			#update y axis limit
			#print("reached readin\n")
			#ymin = (min(ydata))-10
			#ymax = (max(ydata))+10
			#plot.ylim([ymin,ymax])
			#ydata.append(readin)
			#del ydata[0]
			xs.append(count)
			ys.append(readin) # update the data
			count = count + 1
			#del ys[0]
			ax1.clear()
			ax1.plot(xs, ys)
		#time.sleep(.500)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()