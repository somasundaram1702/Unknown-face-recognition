
from detect_face_video import detect_face
import cv2
import pickle
import numpy as np
import argparse


class inference():

  def load_model(self):

      with open('EVM_model.pkl','rb') as b:
          self.mevm = pickle.load(b)

      with open('class_names.pkl','rb') as cl:
          self.cls_names = pickle.load(cl)

  def read_video(self,path):

      self.cap = cv2.VideoCapture(path)
      width, height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
      fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
      #fourcc = cv2.cv.CV_FOURCC(*'XVID')
      self.videowriter = cv2.VideoWriter('output_video.avi', fourcc, 20, (int(width),int(height),))

      #Facetection class to be loaded
      self.df = detect_face()
      self.df.load_facenet()

  def run_inference(self):

        while(self.cap.isOpened()):

            # Capture frame-by-frame
            ret, frame0 = self.cap.read()
            if ret == True:
              try:
                  frame = cv2.cvtColor(frame0,cv2.COLOR_BGR2RGB)
                  #frame = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  res = self.df.vid_detect_face(frame)
                  bb_img = self.df.draw_detect(frame)
                  face = self.df.vid_crop_resize()
                  #print(len(res))
                  #print(np.shape(face))
                  for i in range(len(res)):
                    embds = self.df.get_embedding(np.array(face[i]))
                    print(np.shape(embds))
                    x = self.df.normalize([embds])
                    probs,index = self.mevm.max_probabilities(x)
                    #bb_img = bb_img[:,:,::-1]
                    print(probs,self.cls_names[index[0][0]]) 
                    if probs[0] > 0.9:
                        cv2.putText(bb_img,self.cls_names[index[0][0]],(res[i]['box'][0],res[i]['box'][1]-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
                        self.videowriter.write(bb_img[:,:,::-1])
                    if probs[0] < 0.2:
                        cv2.putText(bb_img,"unknown",(res[i]['box'][0],res[i]['box'][1]-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
                        self.videowriter.write(bb_img[:,:,::-1])
              except Exception as e:
                  print(e)
                  self.videowriter.write(frame)
                  continue
            else:
              break

        self.videowriter.release()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("args.video_path")
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',required=True,
                          help = 'Please input the path of the input video')
    args = parser.parse_args()

    inf = inference()
    inf.load_model()
    
    inf.read_video(args.video_path)
    inf.run_inference()

    
