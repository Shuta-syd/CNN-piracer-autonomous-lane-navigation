import socket
import cv2
import numpy as np
import logging
import struct
from keras.models import load_model
import tensorflow as tf

MAX_DGRAM = 2**16
TCP_IP = '192.168.86.34'
TCP_PORT = 5001

def preprocess_image(image):
    height, _, _ = image.shape
    image = image[int(height/5):,:,:]

    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (256, 256))
    image = image / 255
    return image

def predict_direction(model=None, frame=None):
    processed_img = preprocess_image(frame)
    X = np.asarray([processed_img])
    direction_probability = model.predict(X)[0]
    return direction_probability

def dump_buffer(s):
  while True:
    seg, addr = s.recvfrom(MAX_DGRAM)
    print(seg[0])
    if struct.unpack('B', seg[0:1])[0] == 1:
      print('finish emptying buffer')
      break

logging.basicConfig(level=logging.INFO)
logging.info('Lane Navigation Model Loading...')

model = load_model('./model/model1020-VGG16-(256, 256, 3)/lane_navigation_final.h5')

logging.info('Lane Navigation Model Loading Complete')

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((TCP_IP, TCP_PORT))
#s.listen(1)
#print(u'Server socket [ TCP_IP: ' + TCP_IP + ', TCP_PORT: ' + str(TCP_PORT) + ' ] is open')
# conn, addr = s.accept()
#print(u'Server socket is connected with client socket [', addr, u']')

dat = b''
dump_buffer(s)

while True:
  seg, addr = s.recvfrom(MAX_DGRAM)
  if struct.unpack('B', seg[0:1])[0] > 1:
    dat += seg[1:]
  else:
    dat += seg[1:]
  image = cv2.imdecode(np.fromstring(dat, dtype=np.uint8), 1)
  dat = b''

  cv2.imshow('frame', image)
  direction_probability = predict_direction(model, image)
  direction = np.argmax(direction_probability)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  # direction_str = str(direction).encode()
  # conn.send(direction_str)

# conn.close()

# https://millo-l.github.io/Python-Implementing%20TCP%20image%20socket-Server-Client/
# https://medium.com/@fromtheast/fast-camera-live-streaming-with-udp-opencv-de2f84c73562
