#importing dependencies
import cv2
import numpy as np
from glob import glob
import os

#creating directory for storing the generated frames
p='C:/Users/Tanya Joon/Documents/MM 803 Image/UCF-101/*/*.avi'
image_base_checker = glob(p)
for filenames in image_base_checker:
        print(filenames)
        path=str(filenames)+'_frames'
        print(path)        
        access_rights = 0o755

        os.mkdir(path, access_rights)

#creating directory for storing the generated optical flow
image_base_checker = glob(p)
for filenames1 in image_base_checker:
        print(filenames1)
        path1=str(filenames1)+'_flow'
        print(path1)       
        access_rights = 0o755
        os.mkdir(path1, access_rights)

#listing all the videos using glob
image_base_checker = glob(p)
for filenames3 in image_base_checker:
        print(filenames3)
        #extracting the frames from video
        cap = cv2.VideoCapture(filenames3)
        ret, frame1 = cap.read()
        #changing the frame to grayscale
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        i=0
        
        #defining paths to store frames and optical low        
        path=str(filenames3)+'_frames'
        path1=str(filenames3)+'_flow'
        
        
        while(1):
            ret, frame2 = cap.read()
            j=str(i)
            if(ret == False):
                break
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            #generating the optical flow
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            #changing the flow to grayscale
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)


            #saving the generated frames and flow
            file_name = j + '.png'
            ret = cv2.imwrite(os.path.join(path, file_name), frame2)
            print(ret, os.path.join(path, file_name))     
            ret1 = cv2.imwrite(os.path.join(path1, file_name), gray)
            
            #continuing the loop till end of video
            i = i+1
            prvs = next

cap.release()
cv2.destroyAllWindows()