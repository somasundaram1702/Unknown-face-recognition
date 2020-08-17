
#-------------------------------------------------------------------------------------------
#Make sure to load the .h5 facenet file, before running this
#-------------------------------------------------------------------------------------------

import os
import cv2
import time
from mtcnn.mtcnn import MTCNN
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from numpy import savez_compressed
from numpy import load 
from numpy import expand_dims
from numpy import asarray
from sklearn.preprocessing import Normalizer
import argparse

beginn = time.time()

class extract_embeds:
    def __init__(self,path):
        self.detect = MTCNN()
        self.root_path = path
        self.X_train_faces = []
        self.y_train_faces = []
        self.X_test_faces = []
        self.y_test_faces = []
        self.newTrainX = []
        self.newTestX = []

    def identify_face(self,path):
        self.img = cv2.imread(path)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.res = self.detect.detect_faces(self.img)
        return(self.res)
    
    def crop_resize(self):
        x1=self.res[0]['box'][0]
        y1=self.res[0]['box'][1]
        x2=self.res[0]['box'][0]+self.res[0]['box'][2]
        y2=self.res[0]['box'][1]+self.res[0]['box'][3]
        self.face = self.img[y1:y2,x1:x2]
        np.shape(self.face)
        self.face = cv2.resize(self.face,(160,160),cv2.INTER_AREA)
        return(self.face)
        
    def check_folders(self):
        if 'outputs' not in os.listdir():
            os.makedirs('./outputs')

        y,x = sorted(os.listdir(self.root_path),key=len)
        assert(x == 'train' or x == 'Train' or x == 'TRAIN')
        assert(y == 'test' or y == 'Test' or y == 'TEST')
        self.train_path = self.root_path+'/'+x
        self.test_path = self.root_path+'/'+y
        train_classes = os.listdir(self.train_path)
        test_classes = os.listdir(self.test_path)
        try:
            assert(len(train_classes) == len(test_classes))
            print('No. of train classes: {}'.format(len(train_classes)))
            print('No. of test classes: {}'.format(len(test_classes)))
        except Exception as e:
            print('Error: Number of classes in train and test should be same')
    
    def extract_faces(self):
        start1 = time.time()
        print('Inside train folder ....')
        for label in os.listdir(self.train_path): 
            print('Images of '+label + ' under process')
            for pics in os.listdir(self.train_path+'/'+label):
              if pics.endswith('.jpg'):    
                #print('Name of the pic:',pics)
                try:
                  face_details = self.identify_face(self.train_path+'/'+label+'/'+pics)
                  out = self.crop_resize()
                  self.X_train_faces.append(out)
                  self.y_train_faces.append(label)
                except:
                  print('Pic {} in {} is not processed'.format(pics,label))
                  continue
        print('time taken to process {} secs'.format(time.time()-start1))
    
        start2 = time.time()
        print('Inside test folder ....')
        for label in os.listdir(self.test_path):
            
            print('Images of '+label + ' under process')
            for pics in os.listdir(self.test_path+'/'+label):
              if pics.endswith('.jpg') or pics.endswith('.jpeg') or pics.endswith('.png') or pics.endswith('.tif'):
                #print('Name of the pic:',pics)
                try:
                  face_details = self.identify_face(self.test_path+'/'+label+'/'+pics)
                  out = self.crop_resize()
                  self.X_test_faces.append(out)
                  self.y_test_faces.append(label)
                except:
                  print('Pic {} in {} is not processed'.format(pics,label))
                  continue
        print('time taken to process {} secs'.format(time.time()-start2))
        savez_compressed('./outputs/Extracted_faces.npz', self.X_train_faces, self.y_train_faces, self.X_test_faces, self.y_test_faces)
        return(self.X_train_faces, self.y_train_faces, self.X_test_faces, self.y_test_faces)

    def load_facenet(self):
        self.model = load_model('facenet_keras.h5')
        print('Facenet model loaded')  
    
    def get_embedding(self, face_pixels):

        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = self.model.predict(samples)
        return(self.normalize([yhat[0]]))
    
    def normalize(self,data):
        in_encoder = Normalizer(norm='l2')
        data = in_encoder.transform(data)
        return(data)

    def extract_embeddings(self):
        # convert each face in the train set to an embedding
        for face_pixels in self.X_train_faces:
            embedding = self.get_embedding(face_pixels)
            self.newTrainX.append(embedding)
        self.newTrainX = asarray(self.newTrainX)
        print(self.newTrainX.shape)

        # convert each face in the test set to an embedding 
        for face_pixels in self.X_test_faces:
            embedding = self.get_embedding(face_pixels)
            self.newTestX.append(embedding)
        self.newTestX = asarray(self.newTestX)
        print(self.newTestX.shape)                                      
        # save arrays to one file in compressed format
        savez_compressed('./outputs/Extracted_embeddings.npz', self.newTrainX, self.y_train_faces, self.newTestX, self.y_test_faces)
        print('Time taken from reading images to extracting embeds: {} s'.format(time.time()-beginn))
  
if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',default=None, required = True, 
                        help = 'Enter the path of the folder contains train and test images')
    args=parser.parse_args()
    
    _extract_embeds = extract_embeds(args.input_path)
    _extract_embeds.check_folders()
    x,y,a,b = _extract_embeds.extract_faces()
    _extract_embeds.load_facenet()
    _extract_embeds.extract_embeddings()
