import cv2
import numpy as np
import time
import logging
from keras.models import load_model
import tensorflow as tf
from piracer import vehicles, control_piracer

DIRECTION = ["LEFT", "STRAIGHT", "RIGHT"]

def preprocess_image(image):
    height, _, _ = image.shape
    image = image[int(height/5):,:,:]

    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image

def predict_direction(model=None, frame=None):
    processed_img = preprocess_image(frame)
    X = np.asarray([processed_img])
    direction_probability = model.predict(X)[0]
    return direction_probability

if __name__ == '__main__':
  try:
    logging.basicConfig(level=logging.INFO)
    logging.info('Lane Navigation Model Loading...')

    model = load_model('model/model-nvdia-(240,180,3)-5class/lane_navigation_final.h5')

    logging.info('Lane Navigation Model Loading Complete')

    vehicle = vehicles.PiRacerStandard()
    vehicle.set_throttle_percent(0.2)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    frame_cnt = -1
    PREDICT_LIMITATION = 4
    while(cap.isOpened()):
      frame_cnt += 1
      if frame_cnt > 20:
          frame_cnt = -1
      if frame_cnt % PREDICT_LIMITATION != 0:
          continue

      ret, original_frame = cap.read()
      if ret == False:
        break

      if frame_cnt % PREDICT_LIMITATION == 0:
          frame = cv2.flip(original_frame, -1)
          direction_probability = predict_direction(model, frame)
          direction_index = np.argmax(direction_probability)
          print(f'direction: {direction_index} ', DIRECTION[direction_index])
          control_piracer.control(piracer=vehicle, direction=direction_index)
  finally:
    cap.release()
    vehicle.set_throttle_percent(0)
    vehicle.set_steering_percent(0)
