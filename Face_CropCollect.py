import cv2 as cv
import numpy as np

#얼굴 인식용 xml 파일
face_classifier = cv.CascadeClassifier("datas/haar_cascade_files/haarcascade_frontalface_default.xml")


def face_extractor(img): #전체 사진에서 얼굴 부위만 잘라 리턴

    img = cv.resize(img, dsize=(640, 480))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  #흑백처리 
    
    faces = face_classifier.detectMultiScale(gray,1.3,5) #얼굴 찾기 

    if faces == (): #찾은 얼굴이 없으면 None으로 리턴
        return None

    for(x,y,w,h) in faces: #얼굴들이 있으면 해당 얼굴 크기만큼 cropped_face에 잘라 넣기
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face #cropped_face 리턴

count = 0 #저장할 이미지 카운트 변수

import os
directory_name = 'datas/images/faces/' #이미지 저장할 디렉토리 지정

if not os.path.exists(directory_name): #위의 디렉토리가 없으면 생성
    os.mkdir(directory_name)
src = "bts 정국/Son-Yeon-Jae.jpg"


while True:
    img = cv.imread(src)                 # 카메라로 부터 사진 1장 얻기 
    cropped_area = face_extractor(img)    # 얼굴 감지 하여 얼굴만 가져오기 
    put_text = ''                           # 윈도우에 표시될 str 변수
    
    if cropped_area is not None:                    #얼굴 감지 하여 얼굴만 가져오기
        count+=1                                    #카운트 하면서
        area = cv.resize(cropped_area,(200,200))    #얼굴 이미지 크기를 200x200으로 조정
        area = cv.cvtColor(area, cv.COLOR_BGR2GRAY) #조정된 이미지를 흑백으로 변환
        
        file_name = directory_name + 'user'+str(count)+'.jpg' #파일이름 생성
        cv.imwrite(file_name,area)                            #저장 실행  

        put_text = "Face Found!, imwrite() count" + str(count) #저장 실행 표시할 변수
    else:
        put_text = "Face not Found" #예외처리

    cv.putText(area,put_text,(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) # 저장 실행 표시
    cv.imshow('Face Cropper',area) # 화면에 얼굴 표시                                 

    if cv.waitKey(1)==ord('q') or count==50:     # 이미지 1000개 획득 혹은 q누르고 종료
        break

cv.destroyAllWindows()  #화면 종료
print('Colleting Samples Complete!!!') #CODE 정상 실행 완료 확인