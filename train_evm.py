

##------------------------------------------------------------------------------------------------------
# In this code, we are trying to Fit EVM and check the accuracy of fitting all classes
# Any number of classes can be used. It generates, fits each class with label and prints accuracy
# At present this code is used to check only, the known classes
#-------------------------------------------------------------------------------------------------------

import numpy as np
from numpy import load
import EVM, scipy
import time
from sklearn.metrics import accuracy_score
import argparse
import pickle


class evm_acc:

    def __init__(self,embeds_path):

      self.trainX={}
      self.testX = {}
      self.change_test=[]
      self.change_train=[]
      self.predicted_cls_nam = []
      self.embeds_path = embeds_path
      data = load(self.embeds_path)
      #self.tail_size = tail_size
      self.trainx,self.trainy,self.testx,self.testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    
    def class_split(self):

        self.change_test = [i+1 for i in range(len(self.testy)-1) if (self.testy[i] != self.testy[i+1])]
        self.change_train = [i+1 for i in range(len(self.trainy)-1) if (self.trainy[i] != self.trainy[i+1])]

        for i in range(len(self.change_test)):
          if i ==0:
            #self.testX['class_{}'.format(i)] = self.testx[i]
            self.testX['class_{}'.format(i)] = self.testx[i:self.change_test[i]]
          else:
            self.testX['class_{}'.format(i)] = self.testx[self.change_test[i-1]:self.change_test[i]]
          #print(self.testX['class_{}'.format(i)].shape)
        self.testX['class_{}'.format(i+1)] = self.testx[self.change_test[i]:]
        #print(self.testX['class_3'].shape)

        for i in range(len(self.change_train)):
          #print(i)
          if i ==0:
            #self.trainX['class_{}'.format(i)] = self.trainx[i]
            self.trainX['class_{}'.format(i)] = self.trainx[i:self.change_train[i]]
          else:
            self.trainX['class_{}'.format(i)] = self.trainx[self.change_train[i-1]:self.change_train[i]]
          #print(self.trainX['class_{}'.format(i)].shape)
        self.trainX['class_{}'.format(i+1)] = self.trainx[self.change_train[i]:]
        #print(self.trainX['class_3'].shape)

    def fit_evm(self,ts,dm,thresh): 
        print('Fitting the EVM Model .....')
        self.ts = int(ts)
        self.dm = float(dm)
        self.thresh = float(thresh)
        self.mevm = EVM.MultipleEVM(tailsize=self.ts, cover_threshold =self.thresh, distance_multiplier = self.dm, distance_function = scipy.spatial.distance.euclidean)
        self.mevm.train([self.trainX[i] for i in list(self.trainX.keys())])

    def extract_class(self):
      self.indexes = np.unique(self.testy, return_index=True)[1]
      self.uni_classes = [self.testy[index] for index in sorted(self.indexes)]
    
    def test_evm(self):
        for i in range(len(self.testx)):
          self.probs,self.index = self.mevm.max_probabilities([self.testx[i]])
          #print(self.probs,self.index)
          #print(self.probs,self.index,i)
          #print('original',self.testy[i])
          #if self.probs[0] > 0.6:
          self.predicted_cls_nam.append(self.uni_classes[self.index[0][0]])
            #print('class_nam',cls_nam)

    def check_accuracy(self):

        score = accuracy_score(self.testy,self.predicted_cls_nam) 
        print('Distance multiplier {}, cover_threshold {}, tail_size {} and Accuracy {}'.format(self.dm,self.thresh,self.ts,score))
        #for i in range(len(testy)):
          #print(self.testy[i],self.predicted_cls_nam[i])

    def save_model(self):

        with open('./outputs/EVM_model.pkl','wb') as f:
          pickle.dump(self.mevm,f)

        with open('./outputs/class_names.pkl','wb') as t:
            pickle.dump(self.uni_classes,t)

if __name__ == '__main__':

      parser = argparse.ArgumentParser()
      parser.add_argument('--Embeds_path',default=None,required=True,
                          help = "Enter the path of the Extracted embeddings")
      parser.add_argument('--tail_size', default=300,
                          help = "Hyperparameter tail size for EVM model")
      parser.add_argument('--threshold', default=0.7,
                          help = "Hyperparameter threshold value for EVM model")
      parser.add_argument('--dist_multiplier', default=0.7,
                          help = "Hyperparameter distance multiplier for EVM model")
      args = parser.parse_args()


      
      sp = evm_acc(args.Embeds_path)
      sp.class_split()
      sp.fit_evm(args.tail_size,args.threshold,args.dist_multiplier)
      sp.extract_class()
      sp.test_evm() 
      sp.check_accuracy()
      sp.save_model()
      
      

