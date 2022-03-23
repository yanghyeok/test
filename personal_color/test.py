from cgitb import grey
from re import S
from turtle import shape
# from cv2 import waitKey
import dlib 
import cv2
import numpy as np
from imutils import face_utils

# 얼굴인식
detector = dlib.get_frontal_face_detector()
# 얼굴의 68개 landmark 찾기

# Mac
# predictor = dlib.shape_predictor('/Users/yanghyeok/Desktop/ShowMeTheColor-master/res/shape_predictor_68_face_landmarks.dat')
# Window
predictor = dlib.shape_predictor('C:/Users/m2appl/Desktop/1/res/shape_predictor_68_face_landmarks.dat')

#face detection part

# Mac
# img_path = '/Users/yanghyeok/Desktop/ShowMeTheColor-master/res/test/nfall/1.jpg'
# Window
img_path = "C:/Users/m2appl/Desktop/1/res/test/nfall/3.jpg"

img = cv2.imread(img_path) #이미지를 컬러로 읽어옴
# 이미지 읽어오기
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img.shape) (500, 485, 3) -> 3 : rgb
# print(len(img)) #세로 500
# print(len(img[0])) #가로 485

# init face parts
right_eyebrow = []
left_eyebrow = []
right_eye = []
left_eye = []
left_cheek = []
right_cheek = []

face_parts = [[],[],[],[],[],[],[],[]]
# 이미지 그레이 스케일로 변환 -> 컬러영상은 3배많은 메모리를 필요로하기 때문
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rect = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
# print(rect) -> rectangles[[(98, 27) (419, 348)]]
# cv2.imshow("grey", grey)
# cv2.waitKey()
faces = rect

for face in faces: 
    # 랜드마크 검출 
    lm = predictor(img, face) 
    lm_point = [] 
    for p in lm.parts(): 
        lm_point.append([p.x, p.y]) 
    lm_point = np.array(lm_point) 
        
    for p in lm_point: 
        cv2.circle(img, (p[0], p[1]), radius=2, color=(255,0,0), thickness=2)

# cv2.imshow("a",cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), color=(0,255,0), thickness=2))
# cv2.waitKey()
# cv2.destroyAllWindows()

rect_1 = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)[0]
shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rect_1)
shape = face_utils.shape_to_np(shape)

idx = 0
for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            face_parts[idx] = shape[i:j]
            idx += 1
face_parts = face_parts[1:5]
print(face_parts[1])