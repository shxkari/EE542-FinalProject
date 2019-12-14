import serial 
from serial import Serial
import numpy as np
from matplotlib import pyplot as plot
from matplotlib.animation import FuncAnimation
from matplotlib import style
import csv
import sys

device_name = '/dev/ttyXRUSB0'
ser = serial.Serial(device_name)

ser.baudrate = 115200
ser.close()
ser.open()

#set plot to animaated
plot.ion()
fig = plot.figure()

ydata = [0]*50
ax1 = plot.axes()
line, = plot.plot(ydata)
plot.ylim ([0,32000]) #sets y range from 10 to 40
<<<<<<< HEAD
#plot.style.use('fivethirtyeight')
#def animate(i):
=======

>>>>>>> e0f04b11bbe133f15f2e4527007eaec6be9e9f7d
while True:
	print("reached while\n")
	# Read a line and strip whitespace from it
	readin = ser.readline().rstrip()
	if readin:  # If it isn't a blank line
		#update y axis limit
		print("reached readin\n")
		#ymin = (min(ydata))-10
		#ymax = (max(ydata))+10
		#plot.ylim([ymin,ymax])
		ydata.append(readin)
		del ydata[0]
		line.set_xdata(np.arange(len(ydata)))
		line.set_ydata(ydata) # update the data
		#update the plot

		plot.draw() 
		plot.show()
#ani = FuncAnimation(plot.gcf(), animate, 1000)
#plot.tight_layout()
#plot.show()