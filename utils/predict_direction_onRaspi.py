import socket
import cv2
import numpy as np
from piracer import vehicles, control_piracer
import base64

TCP_SERVER_IP = '192.168.86.34'
TCP_SERVER_PORT = 5001
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(TCP_SERVER_PORT) + ' ]')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  340)

fps = 20
predict_timing = True
while True:
  if fps == 0:
    fps = 20
    predict_timing = True
  elif fps == False:
    continue;

  vehicle = vehicles.PiRacerStandard()
  ret, original_frame = cap.read()
  if ret == False:
    break
  frame = cv2.flip(original_frame, -1)

  encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
  result, imgencode = cv2.imencode('.jpg', frame, encode_param)
  data = np.array(imgencode)
  stringData = base64.b64encode(data)
  length = str(len(stringData))

  s.sendall(length.encode('utf-8').ljust(64))
  s.send(stringData)

  direction_str = s.recv(1024)
  direction = int(direction_str.decode())
  print('Direction: ', direction)
  control_piracer.control(piracer=vehicle, direction=direction)
  fps = fps - 1

cap.release()
cv2.destroyAllWindows()
s.close()
