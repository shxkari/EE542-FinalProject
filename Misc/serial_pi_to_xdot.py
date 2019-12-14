import time
import serial
import sys
print (sys.argv)

device = '/dev/ttyACM1'
ser = serial.Serial(device)
ser.baudrate = 9600

ser.flushOutput()
ser.flushInput()

counter = 1

while 1:
	line = "not"
	while "ready" not in line:
		line = ser.readline()
		line = line.decode('ascii')
	data = input("Data: ")  # Python 3
	max_val = input("Max Val: ")
	text = "D:" + data + ",M:" + max_val + '!'
	# hi = str(counter) + "\n"
	ser.write(text.encode())
	print(text)
	print(text.encode())

