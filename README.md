### Create_Dataset_Face
## Dataset을 직접 만들어서 머신러닝으로 학습시킨 OCR 프로그램

1. 데이터 수집
  Scraping_faces_google.py (아무사진 스크래핑용)
  Door_locker_OCR.py (내 사진 촬영용)
  
  내 사진 : 1000 장
  아무 사진 : 1000 장
  
2. 전처리
  수집된 아무사진을
  convert_images.py로 사진을 전처리하여
  data > 0 에 저장된다.
    
3. CreateDataset.py
  dataset을 만드는 코드.
  
  수집/전처리된 사진들의
  
  최종적인 폴더 배치는
  dataset > 1 : 내 사진 1000장
  dataset > 0 : 아무사진 1000장
  
  (두 폴더의 사진숫자는 동일해야 한다.)  

4. MLTrainer.py
  직접 만든 dataset을 import하여 머신러닝으로 LogisticRegression()함수로 학습하여
  모델 matrix.pkl 을 생성함.
  
5. Myface_detector.py (메인 프로그램 )
  모델 matrix.pkl을 load하여 model.predict()함수로 나인지(dataset > 1 폴더에 있는 사진 주인공인지),
  아닌지 판별하는 프로그램.
  
  웹캠 인식 오류는...
  cap = cv.VideoCapture(0) # 괄호안에 (0) 자신의 컴퓨터의 웹캠 번호, 모르겠으면, 0,1,2,3... 해보면 인식된다. 
