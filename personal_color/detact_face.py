from tkinter import Image
from turtle import shape
import dlib 
import cv2 as cv 
import numpy as np
from PIL import Image

face_det = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("/Users/yanghyeok/Desktop/shape_predictor_68_face_landmarks.dat")

right_eyebrow = []
left_eyebrow = []
right_eye = []
left_eye = []
left_cheek = []
right_cheek = []


face_parts = [[],[],[],[],[],[],[],[]]
# detect faces in the grayscale image
rect = detector(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 1)[0]
# determine the facial landmarks for the face region, then
# convert the landmark (x, y)-coordinates to a NumPy array
shape = self.predictor(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), rect)
shape = face_utils.shape_to_np(shape)


if __name__ == "__main__": 
    src = cv.imread("/Users/yanghyeok/Desktop/ShowMeTheColor-master/res/test/nfall/1.jpg") 
    grey = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # 얼굴검출 
    faces = face_det(grey) 

    img_name = "/Users/yanghyeok/Desktop/ShowMeTheColor-master/res/test/nfall/1.jpg"
    im = Image.open(img_name)
    pix = np.array(im)

    for face in faces: # 랜드마크 검출 
        lm = landmark_model(src, face) 
        lm_point = [] 
    for p in lm.parts(): 
        lm_point.append([p.x, p.y]) 
    lm_point = np.array(lm_point) 
    
    print(lm_point)
    
    for p in lm_point:
        cv.circle(src, (p[0], p[1]), radius=2, color=(255,0,0), thickness=2) 
    cv.imshow("lsh", src) 
    cv.waitKey() 
    cv.destroyAllWindows()
