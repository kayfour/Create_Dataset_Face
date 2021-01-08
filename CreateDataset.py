#~$ vi CreateDataset.py
import os
import cv2 as cv
import numpy as np
# from sklearn.model_selection import train_test_split
# import re, glob 

def create_dataset(img_folder):
    img_data_array = []; class_name = []
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv.imread(image_path, cv.COLOR_BGR2RGB)
            image = np.array(image).astype('float32')
            image /= 255
            img_data_array.append(image.reshape(-1))            
            class_name.append(dir1)
    return img_data_array, class_name
#img_data, class_name = create_dataset('dataset')




"""
#현재 로컬 이미지 폴더 구조
#dataset/25/road, water, building, green

imagePath = './dataset/25'
categories = ["road", "water", "building", "green"]

#dataset/25 하위 폴더의 이름이 카테고리가 됨. 동일하게 맞춰줘야한다.
nb_classes = len(categories)  

image_w = 28 
image_h = 28 

X = []  
Y = []  

for idx, cate in enumerate(categories):  
    label = [0 for i in range(nb_classes)]  
    label[idx] = 1  
    image_dir = imagePath+'/'+cate+'/'  
     
    for top, dir, f in os.walk(image_dir): 
        for filename in f:  
            print(image_dir+filename)  
            img = cv2.imread(image_dir+filename)  
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])  
            X.append(img/256)  
            Y.append(label)  
             
X = np.array(X)  
Y = np.array(Y)  

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)  
xy = (X_train, X_test, Y_train, Y_test) 
 

#생성된 데이터셋을 저장할 경로와 파일이름 지정
np.save("./imageDataList_25.npy", xy)
"""