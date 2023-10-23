# - import python library
import os
import cv2
import time

from color_code.color import *

# - variables
index = 1
VIDEO = 0
DATASET = "dataset/"
TERM_SIZE = os.get_terminal_size().columns

# - capture img program
def record_data(vehicle):
    print(
        f"{CYA}{BOL}[INFORMT]{RES}    ",
        f"Capture-img process has been started at:",
        "\n",
        f"{CYA}{BOL}         {RES}    ",
        f"{time.time()}"
    )

    # Start video capture
    cap = cv2.VideoCapture(VIDEO)

    # Verify that the camera is available
    if not cap.isOpened():
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Can't open camera. Please check the connection!"
        )
        return

    try:
        while True:
            rst, frame  = cap.read()
            frame       = cv2.flip(frame, -1)
            steering    = vehicle.get_steering_raw_data()
            direction   = 1
            if (steering < 3000):
                direction = 2
            elif (steering > 5500):
                direction = 3

            if not rst:
                print(
                    f"{RED}{BOL}[FAILURE]{RES}    ",
                    f"Can't read the frame.",
                    "\n",
                    f"{RED}{BOL}         {RES}    ",
                    f"Please follow steps to solve the problem.",
                    "\n",
                    f"{RED}{BOL}         {RES}    ",
                    f" - Check camera connection and drivers.",
                    "\n",
                    f"{RED}{BOL}         {RES}    ",
                    f" - Check is another application using camera."
                )
                break

            cv2.imwrite(f'{DATASET}/frames/frame_{index:04d}_{direction:04d}.jpg', frame)
            index += 1

    except Exception as exception:
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Unexpected exception has occured.\n",
            '-'*TERM_SIZE, "\n",
            exception, "\n",
            '-'*TERM_SIZE,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()
