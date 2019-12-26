# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:17:25 2019

@author: Tanya Joon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:45:07 2019

@author: Tanya Joon
"""

from glob import glob
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Average, Dropout, UpSampling2D, Conv2DTranspose, Reshape
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from tensorflow.keras import models
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

BATCH_SIZE = 16

def datagen():
    
    while True:
        
        samples = np.random.randint(0, len(files_frames), size = BATCH_SIZE)
        #yield [np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]), np.array([np.reshape(cv2.resize(cv2.imread(file, 0), (224, 224)), (224, 224, 1)) for file in files_flow[samples]])], to_categorical([dct[file.split('/')[6]]-1 for file in files_frames[samples]], 101)
        yield np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]), [np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]),
             np.array([np.reshape(cv2.resize(cv2.imread(file, 0), (224, 224)), (224, 224, 1)) for file in files_flow[samples]]), to_categorical([dct[file.split('/')[6]]-1 for file in files_frames[samples]], 101)]

gen = datagen()

inp_frames = Input(shape=(224,224,3))
#Layer1
conv_1_frames = Conv2D(32, (3,3), activation='relu',padding='same')(inp_frames)
batch_norm_1_frames= tf.nn.local_response_normalization(conv_1_frames, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
pool_1_frames = MaxPooling2D((2,2)) (batch_norm_1_frames)
#Layer2
conv_2_frames = Conv2D(32, (3,3), activation='relu',padding='same')(pool_1_frames)
batch_norm_2_frames = tf.nn.local_response_normalization(conv_2_frames, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
pool_2_frames = MaxPooling2D((2,2)) (batch_norm_2_frames)
#Layer3
conv_3_frames = Conv2D(32,(3,3),activation='relu',padding='same')(pool_2_frames)
#Layer4
conv_4_frames = Conv2D(32,(3,3),activation='relu',padding='same')(conv_3_frames)
#Layer5
conv_5_frames = Conv2D(32,(3,3),activation='relu',padding='same')(conv_4_frames)
pool_3_frames = MaxPooling2D((2,2))(conv_5_frames)

#Frames Decoder
pool_3_de_frames = UpSampling2D((2,2)) (pool_3_frames)
conv_5_de_frames = Conv2DTranspose(32,(3,3),activation='relu',padding='same') (pool_3_de_frames)
conv_4_de_frames = Conv2DTranspose(32,(3,3),activation='relu',padding='same') (conv_5_de_frames)
conv_3_de_frames = Conv2DTranspose(32,(3,3),activation='relu',padding='same') (conv_4_de_frames)
pool_2_de_frames = UpSampling2D((2,2)) (conv_3_de_frames)
conv_2_de_frames = Conv2DTranspose(32, (3,3), activation='relu',padding='same') (pool_2_de_frames)
batch_norm_2_frames = tf.nn.local_response_normalization(conv_2_de_frames, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
#conv_3_de_frames = Conv2DTranspose(3,)

pool_1_de_frames = UpSampling2D((2,2)) (batch_norm_2_frames)
conv_1_de_frames = Conv2DTranspose(32, (3,3), activation='relu',padding='same') (pool_1_de_frames)
batch_norm_1_frames= tf.nn.local_response_normalization(conv_1_de_frames, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
conv_1 = Conv2DTranspose(3,(3,3), activation = 'relu', padding = 'same')(batch_norm_1_frames)
#Flow Decoder
pool_3_de_flow = UpSampling2D((2,2)) (pool_3_frames)
conv_5_de_flow = Conv2DTranspose(32,(3,3),activation='relu',padding='same') (pool_3_de_flow)
conv_4_de_flow = Conv2DTranspose(32,(3,3),activation='relu',padding='same') (conv_5_de_flow)
conv_3_de_flow = Conv2DTranspose(32,(3,3),activation='relu',padding='same') (conv_4_de_flow)
pool_2_de_flow = UpSampling2D((2,2)) (conv_3_de_flow)

conv_2_de_flow = Conv2DTranspose(32, (3,3), activation='relu',padding='same') (pool_2_de_flow)
pool_1_de_flow = UpSampling2D((2,2)) (conv_2_de_flow)
conv_1_de_flow = Conv2DTranspose(32, (3,3), activation='relu',padding='same') (pool_1_de_flow)
batch_norm_1_flow= tf.nn.local_response_normalization(conv_1_de_flow, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
conv_2 = Conv2DTranspose(1,(3,3), activation = 'relu', padding = 'same')(batch_norm_1_flow)

#reshape2= Reshape((224,224)) (batch_norm_1_flow)
#output_flow= Output(shape=(224,224,3))(conv_1_de_flow)

#Label decoder
flat_frames = Flatten() (pool_3_frames)
#Layer6
fc_1_frames = Dense(2048,activation='relu')(flat_frames)
#drop_1 = Dropout(0.1)(fc_1)
#Layer7
fc_2_frames = Dense(2048,activation='relu')(fc_1_frames)

#drop_2 = Dropout(0.1)(fc_2)
out_label = Dense(101,activation='softmax')(fc_2_frames)

#model = Model(inputs=inp_frames, outputs=[conv_1,conv_2,out_label])


model = models.load_model('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model_ed_x_8.h5')
print(model)
model.summary()

opti_flow = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss_list=['mse','mse','categorical_crossentropy']
model.compile(optimizer=opti_flow, loss = loss_list, metrics=['mae','accuracy'])

tensorboard = TensorBoard(log_dir="logs\{}".format(time()),write_graph=True)

i=0
hist_frames=[]
for x, y in datagen():
    i=i+1 
    print(i)
    if(i == 5000): break
    history = model.fit(x,y, batch_size=16, epochs=1)
    hist_frames.append(history.history) 
    if(i%100==0):
            model.save('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model_ed_x_9.h5')

        
        
print("\nhistory dict:",hist_frames)

model.save('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model_ed_x_9.h5')

loss_array=[]
for i in hist_frames:
    for j in i['dense_2_loss']:
        loss_array.append(j)
        
accuracy_array=[]
for i in hist_frames:
    for j in i['dense_2_accuracy']:
        accuracy_array.append(j)   
        


print("Average loss: ", np.average(loss_array))

print("Average accuracy: ", np.average(accuracy_array)) 
                


plt.plot(accuracy_array[0::50])
#plt.plot(mae_array[0::200])
plt.title('model accuracy')
plt.ylabel('')
plt.xlabel('')
plt.legend(['accuracy'], loc='upper left')
#plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_ed_x_9.png')
#plt.show()
# summarize history for loss
plt.clf()  
plt.plot(loss_array[0::50])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('')
#plt.show()
plt.savefig('loss_ed_x_9.png')
#for x, y in datagen():
#	model.train_on_batch(x,y)
   