chunk = 1024
with open("test", 'rb') as f:
    while(True):
        data = f.read(chunk)
        print("read {} bytes".format(chunk))

