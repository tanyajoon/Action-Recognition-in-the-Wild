# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:07:00 2019

@author: Tanya Joon
"""

#importing dependencies
import tensorflow as tf 
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.utils import to_categorical
from glob import glob
import cv2
import matplotlib.pyplot as plt

#loading the saved model
model = models.load_model('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model_10000_saved.h5')
print(model)

#compiling the model by using categorical_crossentropy loss
opti = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opti, loss = 'categorical_crossentropy', metrics=['mse','accuracy'])
model.summary()

#path to the testing list
path='C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\split\\testlist.txt'
f = open(path)
line = f.readline()
path1= 'C:/Users/Tanya Joon/Documents/MM 803 Image/UCF-101/'
i = 0
w = [];
files_frames = []
files_flow = []
while line:
    #reading the training list line by line and appending the frames and optical flow names to the list
    line1=line[:-1]
    filename_frames= path1+line1+'_frames'  
    filename_flow=path1+line1+'_flow' 
    line = f.readline()
    imagePatches_frames = glob(filename_frames+'/*.png')
    imagePatches_flow = glob(filename_flow+'/*.png') 
    count_frames=0;
    count_flow=0;
    files_frames += imagePatches_frames
    files_flow += imagePatches_flow
    count_frames += len(imagePatches_frames)
    count_flow += len(imagePatches_flow)
print(count_frames)
print(count_flow)
f.close()

#storing the frames and optical flow names to a numpy array
files_frames = np.array(files_frames)
files_flow = np.array(files_flow)

#reading all the 101 class labels from the file
dct = {}
for line in open("C:/Users/Tanya Joon/Documents/MM 803 Image/split/classInd.txt", "r").readlines():
    x, y = line.strip().split(' ')
    dct[y] = int(x)

BATCH_SIZE = 16

#defining a data generator to feed the frames and optical flow to the model
def datagen():
    
    while True:
        
        samples = np.random.randint(0, len(files_frames), size = BATCH_SIZE)
        yield [np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]), np.array([np.reshape(cv2.resize(cv2.imread(file, 0), (224, 224)), (224, 224, 1)) for file in files_flow[samples]])], to_categorical([dct[file.split('/')[6]]-1 for file in files_frames[samples]], 101)
        
gen = datagen()

#calling the datagenerator and passing the inputs to our model for evaluation
i =0 
hist_frames=[]
for x,y in datagen():
    i = i+1
    if(i == 5000): break
    print(i)
    eval = model.evaluate(x,y,batch_size=16)
    hist_frames.append(eval)
print(hist_frames)
  
#saving the testing loss aand accuracy in an numpy array          
loss_list = [item[0] for item in hist_frames]
accuracy_list= [item[2] for item in hist_frames]

#printing the average testing loss and accuracy   
print("Average loss: ", np.average(loss_list))
print("Average accuracy: ", np.average(accuracy_list))

#visualizing testing accuracy
plt.plot(accuracy_list[0::25])
plt.title('model accuracy')
plt.ylabel('')
plt.xlabel('')
plt.legend(['accuracy'], loc='upper left')
plt.savefig('accuracy_test_ED.png')

#visualizing testing loss
plt.clf()  
plt.plot(loss_list[0::25])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('')
plt.savefig('loss_test_ED.png')
