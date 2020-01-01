# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:29:16 2019

@author: Tanya Joon
"""
#importing dependencies
from glob import glob
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Average
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

#path to the training list
path='C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\split\\trainlist.txt'
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

BATCH_SIZE = 64

#defining a data generator to feed the frames and optical flow to the model
def datagen():
    
    while True:
        
        samples = np.random.randint(0, len(files_frames), size = BATCH_SIZE)
        yield [np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]), np.array([np.reshape(cv2.resize(cv2.imread(file, 0), (224, 224)), (224, 224, 1)) for file in files_flow[samples]])], to_categorical([dct[file.split('/')[6]]-1 for file in files_frames[samples]], 101)
        
gen = datagen()

#Model


#input layer: taking generated frames as input
inp_frames = Input(shape=(224,224,3))
#Layer1
conv_1_frames = Conv2D(96, (7,7), strides= 2, activation='relu')(inp_frames)
batch_norm_1_frames= tf.nn.local_response_normalization(conv_1_frames, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
pool_1_frames = MaxPooling2D((2,2)) (batch_norm_1_frames)
#Layer2
conv_2_frames = Conv2D(256, (5,5), strides= 2, activation='relu')(pool_1_frames)
batch_norm_2_frames = tf.nn.local_response_normalization(conv_2_frames, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
pool_2_frames = MaxPooling2D((2,2)) (batch_norm_2_frames)
#Layer3
conv_3_frames = Conv2D(512,(3,3),strides=1,activation='relu')(pool_2_frames)
#Layer4
conv_4_frames = Conv2D(512,(3,3),strides=1,activation='relu')(conv_3_frames)
#Layer5
conv_5_frames = Conv2D(512,(3,3),strides=1,activation='relu')(conv_4_frames)
pool_3_frames = MaxPooling2D((2,2))(conv_5_frames)
flat_frames = Flatten() (pool_3_frames)
#Layer6
fc_1_frames = Dense(4096,activation='relu')(flat_frames)
#Layer7
fc_2_frames = Dense(2048,activation='relu')(fc_1_frames)
#output layer
out_frames = Dense(101,activation='softmax')(fc_2_frames)


#input layer: taking generated optical flow as input
inp_flow = Input(shape=(224,224,1))
#Layer1
conv_1_flow = Conv2D(96, (7,7), strides= 2, activation='relu')(inp_flow)
batch_norm_1_flow= tf.nn.local_response_normalization(conv_1_flow, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
pool_1_flow = MaxPooling2D((2,2)) (batch_norm_1_flow)
#Layer2
conv_2_flow = Conv2D(256, (5,5), strides= 2, activation='relu')(pool_1_flow)
pool_2_flow = MaxPooling2D((2,2)) (conv_2_flow)
#Layer3
conv_3_flow = Conv2D(512,(3,3),strides=1,activation='relu')(pool_2_flow)
#Layer4
conv_4_flow = Conv2D(512,(3,3),strides=1,activation='relu')(conv_3_flow)
#Layer5
conv_5_flow = Conv2D(512,(3,3),strides=1,activation='relu')(conv_4_flow)
pool_3_flow = MaxPooling2D((2,2))(conv_5_flow)
flat_flow = Flatten() (pool_3_flow)
#Layer6
fc_1_flow = Dense(4096,activation='relu')(flat_flow)
#Layer7
fc_2_flow = Dense(2048,activation='relu')(fc_1_flow)
#output layer
out_flow = Dense(101,activation='softmax')(fc_2_flow)

#Taking the output of both the streams and combining them 
out = Average()([out_frames, out_flow])
model = Model(inputs=[inp_frames, inp_flow], outputs=out)
opti_flow = tf.keras.optimizers.Adam(learning_rate=1e-5)

#compiling the model by using categorical_crossentropy loss
model.compile(optimizer=opti_flow, loss = 'categorical_crossentropy', metrics=['mae','accuracy'])
model.summary()

#visualizing the model on tensorboard
tensorboard = TensorBoard(log_dir="logs\{}".format(time()),write_graph=True)

#calling the datagenerator and passing the inputs to our model for training
i=0
hist_frames=[]
for x, y in datagen():
	i=i+1 
	print(i)
	if(i == 15000): break
	history = model.fit(x,y, batch_size=64, epochs=1,callbacks=[tensorboard])
	hist_frames.append(history.history) 

#saving training history
print("\nhistory dict:",hist_frames)

#saving the model after training
model.save('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model.h5')

#saving the training loss in an numpy array
loss_array=[]
for i in hist_frames:
    for j in i['loss']:
        loss_array.append(j)
        
#saving the training accuracy in an numpy array
accuracy_array=[]
for i in hist_frames:
    for j in i['accuracy']:
        accuracy_array.append(j)   

#printing the average training loss and accuracy                
print("Average accuracy: ", np.average(accuracy_array))
print("Average test loss: ", np.average(loss_array))

#visualizing training accuracy
plt.plot(accuracy_array[0::200])
plt.title('model accuracy')
plt.ylabel('')
plt.xlabel('')
plt.legend(['accuracy'], loc='upper left')
plt.savefig('accuracy.png')

#visualizing training loss
plt.clf()  
plt.plot(loss_array[0::200])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('')
plt.savefig('loss.png')

