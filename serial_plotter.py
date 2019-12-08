import serial 
import numpy as np
from matplotlib import pyplot as plot
import csv
import sys

device_name = '/dev/tty.usbmodem1441202'
ser = serial.Serial(device_name)

ser.baudrate = 115200
ser.open()

#set plot to animaated
plot.ion()

ydata = [0]*50
ax1 = plot.axes()
line, = plot.plot(ydata)
plot.ylim ([0:32000]) #sets y range from 10 to 40

while True:
	# Read a line and strip whitespace from it
	line = ser.readline().strip()
	if line:  # If it isn't a blank line
		#update y axis limit
		ymin = float(min(ydata))-10
		ymax = float(max(ydata))+10
		plt.ylim([ymin,ymax])
		ydata.append(data)
		del ydata[0]
		line.set_xdata(np.arange(len(ydata)))
		line.set_ydata(ydata) # update the data
		#update the plot
		plot.draw() 