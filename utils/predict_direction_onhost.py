import socket
import cv2
import numpy as np
import logging
import base64
from keras.models import load_model
import tensorflow as tf

def preprocess_image(image):
    lower_orange = np.array([0, 40, 40])
    upper_orange = np.array([30, 255, 255])

    height, _, _ = image.shape
    image = image[int(height/5):,:,:]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    masked_img = cv2.inRange(hsv, lower_orange, upper_orange)
    cv2.imshow('mask', masked_img)

    image = cv2.GaussianBlur(masked_img, (3,3), 0)
    image = cv2.resize(image, (256, 256))
    image = image / 255
    return image

def predict_direction(model=None, frame=None):
    processed_img = preprocess_image(frame)
    X = np.asarray([processed_img])
    direction_probability = model.predict(X)[0]
    return direction_probability

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


logging.basicConfig(level=logging.INFO)
logging.info('Lane Navigation Model Loading...')

model = load_model('model/XXX')

logging.info('Lane Navigation Model Loading Complete')

TCP_IP = '192.168.86.34'
TCP_PORT = 5001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print(u'Server socket [ TCP_IP: ' + TCP_IP + ', TCP_PORT: ' + str(TCP_PORT) + ' ] is open')
conn, addr = s.accept()
print(u'Server socket is connected with client socket [', addr, u']')

while True:
    length = recvall(conn, 64)
    length = length.decode('utf-8')

    stringData = recvall(conn, int(length))

    data = np.frombuffer(base64.b64decode(stringData), np.uint8)
    decimg = cv2.imdecode(data, 1)

    direction_probability = predict_direction(model, decimg)
    left = direction_probability[0]
    center = direction_probability[1]
    right = direction_probability[2]

    if (left > center) and (left > right):
      direction = 1
    elif (right > center) and (right > left):
      direction = 3
    else:
      direction = 2

    direction_str = str(direction).encode()
    conn.send(direction_str)

conn.close()

# https://millo-l.github.io/Python-Implementing%20TCP%20image%20socket-Server-Client/
