import socket
import cv2
import math
import numpy as np
import struct
from piracer import vehicles, control_piracer

MAX_DGRAM = 2**16
MAX_IMAGE_DGRAM = MAX_DGRAM - 64
class FrameSegment(object):
  def __init__(self, sock, port, addr="127.0.0.1") -> None:
    self.s = sock
    self.port = port
    self.addr = addr

  def udp_frame(self, image):
    compress_img = cv2.imencode('.jpg', image)[1]
    dat = compress_img.tostring()
    size = len(dat)
    num_of_segments = math.ceil(size / (MAX_IMAGE_DGRAM))
    array_pos_start = 0

    while num_of_segments:
      array_pos_end = min(size, array_pos_start + MAX_IMAGE_DGRAM)
      self.s.sendto(struct.pack('B', num_of_segments) + dat[array_pos_start:array_pos_end], (self.addr, self.port))
      array_pos_start = array_pos_end
      num_of_segments -= 1

TCP_SERVER_IP = '192.168.86.34'
TCP_SERVER_PORT = 5001
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(TCP_SERVER_PORT) + ' ]')

fs = FrameSegment(s, TCP_SERVER_PORT, TCP_SERVER_IP)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  340)

while True:
  vehicle = vehicles.PiRacerStandard()
  ret, original_frame = cap.read()
  if ret == False:
    break
  frame = cv2.flip(original_frame, -1)
  fs.udp_frame(frame)

  direction_str = s.recv(1024)
  direction = int(direction_str.decode())
  print('Direction: ', direction)
  control_piracer.control(piracer=vehicle, direction=direction)

cap.release()
cv2.destroyAllWindows()
s.close()
