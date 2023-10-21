import cv2
import numpy as np
import math
import logging
from keras.models import load_model
import tensorflow as tf
import piracer


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

def predict_driving(model=None, frame=None):
    processed_img = preprocess_image(frame)
    X = np.asarray([processed_img])
    direction_probability = model.predict(X)[0]
    return direction_probability

if __name__ == '__main__':
  vehicle = piracer.vehicles.PiRacerStandard()

  logging.basicConfig(level=logging.INFO)
  logging.info('Lane Navigation Model Loading...')

  model = load_model('model/XXX')

  logging.info('Lane Navigation Model Loading Complete')

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FPS, 20)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

  while(cap.isOpened()):
    ret, original_frame = cap.read()
    if ret == False:
      break
    frame = cv2.flip(original_frame, -1)
    direction_probability = predict_driving(model, frame)

    left = direction_probability[0]
    center = direction_probability[1]
    right = direction_probability[2]

    if (left > center) and (left > right):
      direction = 1
    elif (right > center) and (right > left):
      direction = 3
    else:
      direction = 2

    print('direction: ', direction)
    piracer.control_piracer.set_steering_direction(piracer=vehicle, direction=direction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()
