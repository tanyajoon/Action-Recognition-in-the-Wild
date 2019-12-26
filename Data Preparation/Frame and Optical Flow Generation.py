import cv2
import numpy as np
from glob import glob
import os

def mkdir_with_mode(directory, mode):
  if not os.path.isdir(directory):
    oldmask = os.umask(0o000)
    os.makedirs(directory, 0o755)
    os.umask(oldmask)

p='C:/Users/Tanya Joon/Documents/MM 803 Image/UCF-101/Archery/*.avi'
image_base_checker = glob(p)
for filenames in image_base_checker:
        print(filenames)
        path=str(filenames)+'_frames'
        print(path)        
        access_rights = 0o755

        os.mkdir(path, access_rights)

image_base_checker = glob(p)
for filenames1 in image_base_checker:
        print(filenames1)
        path1=str(filenames1)+'_flow'
        print(path1)       
        access_rights = 0o755
	os.mkdir(path1, access_rights)


#mkdir_with_mode( path, 0o755 );
#mkdir_with_mode( path1, 0o755 );    


image_base_checker = glob(p)
for filenames3 in image_base_checker:
        print(filenames3)
        cap = cv2.VideoCapture(filenames3)
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        ret, frame2 = cap.read()
        
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        i=0
        
        path=str(filenames3)+'_frames'
        path1=str(filenames3)+'_flow'
        
        
        while(1):
            ret, frame2 = cap.read()
            j=str(i)
            if(ret == False):
                break
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

            print(path)
           
            
            
            file_name = j + '.png'
            ret = cv2.imwrite(os.path.join(path, file_name), frame2)
    
            # For debugging.
            print(ret, os.path.join(path, file_name))
            
            ret1 = cv2.imwrite(os.path.join(path1, file_name), gray)
            #cv2.imwrite(os.path.join(path , 'j',frame2)
            #cv2.imwrite(os.path.join(path1 , 'j',gray)
            
            i = i+1
            prvs = next

cap.release()
cv2.destroyAllWindows()