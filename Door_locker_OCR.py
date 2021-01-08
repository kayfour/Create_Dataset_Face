from cv2 import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

directory_name = 'datas/images/faces/' #내 사진이 저장될 디렉토리
onlyfiles = [f for f in listdir(directory_name) if isfile(join(directory_name,f))] #faces폴더에 있는 파일 리스트 얻기

Training_Data, Labels = [], []  #데이터와 매칭될 라벨 변수 

for i, files in enumerate(onlyfiles):                   #파일 개수 만큼 루프
    image_path = directory_name + onlyfiles[i]          #불러올 이미지 위치와 이름 변수 지정
    images = cv.imread(image_path, cv.IMREAD_GRAYSCALE) #이미지 불러오기
    Training_Data.append(np.asarray(images, dtype=np.uint8)) #Training_Data 리스트에 이미지를 바이트 배열로 추가 
    Labels.append(i)                                    #Labels 리스트엔 카운트 번호 추가

Labels = np.asarray(Labels, dtype=np.int32) #Labels를 32비트 정수로 변환

model = cv.face.LBPHFaceRecognizer_create() #학습시킬 모델 생성

print("Model Training Start!!!!!")

model.train(np.asarray(Training_Data), np.asarray(Labels)) #학습시작

print("Model Training Complete!!!!!")

face_classifier = cv.CascadeClassifier('datas/haar_cascade_files/haarcascade_frontalface_default.xml')
# #얼굴 인식용 xml 파일 샘플, Haar Cascade 데이터를 로딩

def face_detector(img, size = 0.5):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #흑백 처리
    faces = face_classifier.detectMultiScale(gray,1.3,5) # 얼굴 찾기

    if faces is():                          #찾은 얼굴이 없으면  
        return img,[]                       #img와 빈 리스트 반환
    

    for(x,y,w,h) in faces:                  #찾는 얼굴이 있으면
        cv.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2) # 얼굴에 사각형을 그리자
        roi = img[y:y+h, x:x+w]
        roi = cv.resize(roi, (200,200))

    return img,roi  #검출된 좌표에 사각 박스 그려진 img 와 200,200 사이즈(roi) 반환

cap = cv.VideoCapture(0)  #카메라 열기

while True:

    ret, frame = cap.read() #카메라로 부터 사진 한장 읽기 

    image, face = face_detector(frame) #카메라의 사진에서 얼굴을 추출

    try:
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY) #검출된 사진을 흑백으로 변환
        result = model.predict(face),  # 학습된 모델로 입력된 얼굴을 예측한 결과

        if result[1] < 500: 
            confidence = int(100*(1-(result[1])/300)) # 일치율을 백분률로 변환
            display_string = str(confidence)+'% Confidence , Predict ' # 일치율 화면에 표시 변수
        
        display_string = display_string + str(result[1]) # 일치율 화면에 표시 변수에 result[1]값 같이 표시
        cv.putText(image,display_string,(100,120), cv.FONT_HERSHEY_COMPLEX,1,(250,120,255),2) # 일치율 화면에 표시 


        if confidence > 75: #75% 초과면 동일 인물로 간주해 문이 열린다! 
            cv.putText(image, "Unlocked", (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Face Cropper', image)

        else: #75% 이하면 다른사람으로 간주.. 문은 열리지 않는다!!!
            cv.putText(image, "Locked", (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('Face Cropper', image)


    except:  #얼굴이 검출되지 않을 때 예외처리
        cv.putText(image, "Face Not Found", (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('Face Cropper', image)
        pass

    if cv.waitKey(1)==ord('q'): #프로그램 종료
        break


cap.release()
cv.destroyAllWindows()