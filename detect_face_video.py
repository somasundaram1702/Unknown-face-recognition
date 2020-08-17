
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from numpy import asarray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC




class detect_face:
    def __init__(self):
        self.detect = MTCNN()
        self.img = None
        self.res = None 
        

    
    def vid_detect_face(self,frame):
        self.img_v = frame
        self.res_v = self.detect.detect_faces(frame)
        return(self.res_v)         
    
    def vid_crop_resize(self):
      self.face = []
      for i in range(len(self.res_v)):
        x1=self.res_v[i]['box'][0]
        y1=self.res_v[i]['box'][1]
        x2=self.res_v[i]['box'][0]+self.res_v[i]['box'][2]
        y2=self.res_v[i]['box'][1]+self.res_v[i]['box'][3]
        self._face = self.img_v[y1:y2,x1:x2]
        self._face = cv2.resize(self._face,(160,160),cv2.INTER_AREA)
        self.face.append(self._face)
      return(self.face)   
    
    def load_facenet(self):
        self.model = load_model('facenet_keras.h5')
        print('Facenet model loaded')  
    
    def get_embedding(self, face_pixels):

        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = self.model.predict(samples)
        return yhat[0]
    
    def normalize(self,data):
        in_encoder = Normalizer(norm='l2')
        data = in_encoder.transform(data)
        return(data)

    def draw_detect(self,img):
    
     l=25
     t=2
     for i in range(len(self.res_v)):
        x1 = self.res_v[i]['box'][0]
        y1 = self.res_v[i]['box'][1]
        x2 = self.res_v[i]['box'][0] + self.res_v[i]['box'][2]
        y2 = self.res_v[i]['box'][1] + self.res_v[i]['box'][3]
        
        cv2.line(img,(x1,y1),(x1+l,y1),(255,0,0),t)
        cv2.line(img,(x1,y1),(x1,y1+l),(255,0,0),t)
        
        cv2.line(img,(x2,y1),(x2-l,y1),(255,0,0),t)
        cv2.line(img,(x2,y1),(x2,y1+l),(255,0,0),t)
        
        cv2.line(img,(x2,y2),(x2-l,y2),(255,0,0),t)
        cv2.line(img,(x2,y2),(x2,y2-l),(255,0,0),t)
       
        cv2.line(img,(x1,y2),(x1+l,y2),(255,0,0),t)
        cv2.line(img,(x1,y2),(x1,y2-l),(255,0,0),t)

        #cv2.line(img,(x1[i]+int(l/2),y1[i]+int(b/outer_len)),(x1[i]+int(l/2),y1[i]-int(b/inner_len)),(0,255,255),6)
        #cv2.line(img,(x1[i]+int(l/outer_len),y1[i]+int(b/2)),(x1[i]-int(l/inner_len),y1[i]+int(b/2)),(0,255,255),6)
        #cv2.line(img,(x1[i]+int(l/2),y1[i]+b-int(b/outer_len)),(x1[i]+int(l/2),y1[i]+b+int(b/inner_len)),(0,255,255),6)
        #cv2.line(img,(x1[i]+int(l)-int(l/outer_len),y1[i]+int(b/2)),(x1[i]+l+int(l/inner_len),y1[i]+int(b/2)),(0,255,255),6)
        
    
     return(img)
        
        
