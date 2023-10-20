import cv2
import os
import random
from imgaug import augmenters as img_aug

index = 13

# Define your image processing functions
def blur(image):
    kernel_size = random.randint(1, 5)
    image = cv2.blur(image, (kernel_size, kernel_size))
    return image

def zoom(image):
    zoom = img_aug.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

def pan(image):
    pan = img_aug.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def adjust_brightness(image):
    brightness = img_aug.Multiply((0.7, 1.3))
    image = brightness.augment_image(image)
    return image

# Function to process and rename images
def process_and_rename_images(input_dir, output_dir):
    global index
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, rename_image(filename))
            print(output_path)

            # # Load the image
            image = cv2.imread(input_path)

            # image = blur(image)
            # image = zoom(image)
            # image = pan(image)
            image = adjust_brightness(image)

            cv2.imwrite(output_path, image)

# Function to rename images
def rename_image(filename):
    global index
    parts = filename.split("_")
    fileindex = parts[0].split("frame")[1]
    if fileindex == "01":
        i = index
    else:
        i = index + 10
    parts[0] = "frame" + str(int(i))
    return "_".join(parts)

# Input and output directories
input_directory = "dataset"
output_directory = "dataset_augmented"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process and rename images
process_and_rename_images(input_directory, output_directory)
