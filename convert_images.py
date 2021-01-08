import os
import cv2 as cv
import numpy as np


def convert_images(filename):
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    for dir1 in os.listdir(img_folder):
        count = 1
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1,  file)
            print(image_path)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv.INTER_AREA)
            file_name_path = dataset_path + filename + str(count) + '.jpg'
            cv.imwrite(file_name_path, image)
            print(file_name_path)
            count += 1

if __name__ == "__main__":
    img_folder = "data/"
    dataset_path = "dataset/0/"
    convert_images("dummy")
