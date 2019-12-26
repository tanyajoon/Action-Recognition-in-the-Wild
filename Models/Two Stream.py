# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:29:16 2019

@author: Tanya Joon
"""

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
path='C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\split\\trainlist.txt'

f = open(path)
line = f.readline()
path1= 'C:/Users/Tanya Joon/Documents/MM 803 Image/UCF-101/'
i = 0
w = [];
files_frames = []
files_flow = []
while line:
    #print(line)   
    line1=line[:-1]
    filename_frames= path1+line1+'_frames'  
    filename_flow=path1+line1+'_flow' 
    #print(filename)
    #copyfile('D:\UCF-101\'+ 'ApplyEyeMakeup\v_ApplyEyeMakeup_g08_c01.avi', '/Users/mdjamilurrahman/Downloads/MULTIMEDIA/MM811/Project_Parkinson_Disease/pd_bet/'+ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi)     
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
#D:\UCF-101\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01.avi_flow

files_frames = np.array(files_frames)
files_flow = np.array(files_flow)
#classes = {y: int(x) for x, y in line.strip().split(' ') for line in open("C:/Users/Tanya Joon/Documents/MM 803 Image/split/classInd.txt", "r").readlines()}

dct = {}
for line in open("C:/Users/Tanya Joon/Documents/MM 803 Image/split/classInd.txt", "r").readlines():
    x, y = line.strip().split(' ')
    dct[y] = int(x)

BATCH_SIZE = 64

def datagen():
    
    while True:
        
        samples = np.random.randint(0, len(files_frames), size = BATCH_SIZE)
        yield [np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]), np.array([np.reshape(cv2.resize(cv2.imread(file, 0), (224, 224)), (224, 224, 1)) for file in files_flow[samples]])], to_categorical([dct[file.split('/')[6]]-1 for file in files_frames[samples]], 101)
        
gen = datagen()

def datagen_flow():
    
    while True:
        
        samples_flow = np.random.randint(0, len(files_flow), size = BATCH_SIZE)
        yield np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_flow[samples_flow]]), to_categorical([dct[file.split('/')[6]]-1 for file in files_flow[samples_flow]], 101)
        
#generator

#for x,y in generator:
#   print(y.shape)


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
#drop_1 = Dropout(0.1)(fc_1)
#Layer7
fc_2_frames = Dense(2048,activation='relu')(fc_1_frames)

#drop_2 = Dropout(0.1)(fc_2)
out_frames = Dense(101,activation='softmax')(fc_2_frames)

#model_frames = Model(inputs=inp_frames, outputs=out_frames)
#logits = [16,101]

#loss_function = tf.nn.softmax()
#opti_frames = tf.keras.optimizers.Adam(learning_rate=1e-5)
#model_frames.compile(optimizer=opti_frames, loss = 'categorical_crossentropy', metrics=['mae','accuracy'])

#model_frames.summary()


inp_flow = Input(shape=(224,224,1))
#Layer1
conv_1_flow = Conv2D(96, (7,7), strides= 2, activation='relu')(inp_flow)
batch_norm_1_flow= tf.nn.local_response_normalization(conv_1_flow, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
pool_1_flow = MaxPooling2D((2,2)) (batch_norm_1_flow)
#Layer2
conv_2_flow = Conv2D(256, (5,5), strides= 2, activation='relu')(pool_1_flow)
#batch_norm_2 = tf.nn.local_response_normalization(conv_2, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
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
#drop_1 = Dropout(0.1)(fc_1)
#Layer7
fc_2_flow = Dense(2048,activation='relu')(fc_1_flow)

#drop_2 = Dropout(0.1)(fc_2)
out_flow = Dense(101,activation='softmax')(fc_2_flow)

out = Average()([out_frames, out_flow])

model = Model(inputs=[inp_frames, inp_flow], outputs=out)
#logits = [16,101]

#loss_function = tf.nn.softmax()
opti_flow = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opti_flow, loss = 'categorical_crossentropy', metrics=['mae','accuracy'])

model.summary()

#steps_per_epochs = total_no_of_training_data_points / batch_size  len(x) // 64
#H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#	epochs=EPOCHS)

#History = model.fit_generator(datagen(), steps_per_epoch = 1, epochs = 10)

tensorboard = TensorBoard(log_dir="logs\{}".format(time()),write_graph=True)


i=0
hist_frames=[]
for x, y in datagen():
	i=i+1 
	print(i)
	if(i == 15000): break
	history = model.fit(x,y, batch_size=64, epochs=1,callbacks=[tensorboard])
	hist_frames.append(history.history) 

print("\nhistory dict:",hist_frames)

model.save('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model.h5')

loss_array=[]
for i in hist_frames:
    for j in i['loss']:
        loss_array.append(j)
        
accuracy_array=[]
for i in hist_frames:
    for j in i['accuracy']:
        accuracy_array.append(j)   
        
mae_array=[]
for i in hist_frames:
    for j in i['mae']:
        mae_array.append(j)   
                
print("Average accuracy: ", np.average(accuracy_array))
print("Average mae: ", np.average(mae_array))
print("Average test loss: ", np.average(loss_array))


plt.plot(accuracy_array[0::200])
#plt.plot(mae_array[0::200])
plt.title('model accuracy')
plt.ylabel('')
plt.xlabel('')
plt.legend(['accuracy'], loc='upper left')
#plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
#plt.show()
# summarize history for loss
plt.clf()  
plt.plot(loss_array[0::200])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('')
#plt.show()
plt.savefig('loss.png')
#for x, y in datagen():
#	model.train_on_batch(x,y)

