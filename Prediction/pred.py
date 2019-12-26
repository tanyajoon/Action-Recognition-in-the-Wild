# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:07:00 2019

@author: Tanya Joon
"""
import tensorflow as tf 
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.utils import to_categorical
from glob import glob
import cv2
import matplotlib.pyplot as plt



model = models.load_model('C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\model_10000_saved.h5')
print(model)
opti = tf.keras.optimizers.Adam(learning_rate=1e-5)
#loss_list=['mse','mse','categorical_crossentropy']
model.compile(optimizer=opti, loss = 'categorical_crossentropy', metrics=['mse','accuracy'])

model.summary()

path='C:\\Users\\Tanya Joon\\Documents\\MM 803 Image\\split\\testlist.txt'

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

dct = {}
for line in open("C:/Users/Tanya Joon/Documents/MM 803 Image/split/classInd.txt", "r").readlines():
    x, y = line.strip().split(' ')
    dct[y] = int(x)

BATCH_SIZE = 16

def datagen():
    
    while True:
        
        samples = np.random.randint(0, len(files_frames), size = BATCH_SIZE)
        yield [np.array([cv2.resize(cv2.imread(file), (224, 224)) for file in files_frames[samples]]), np.array([np.reshape(cv2.resize(cv2.imread(file, 0), (224, 224)), (224, 224, 1)) for file in files_flow[samples]])], to_categorical([dct[file.split('/')[6]]-1 for file in files_frames[samples]], 101)
        
gen = datagen()

#for x,y in gen:
#   print(x,y)

#predict_dataset = datagen()

#dict={'ApplyEyeMakeup': 1, 'ApplyLipstick': 2, 'Archery': 3, 'BabyCrawling': 4, 'BalanceBeam': 5, 'BandMarching': 6, 'BaseballPitch': 7, 'Basketball': 8, 'BasketballDunk': 9, 'BenchPress': 10, 'Biking': 11, 'Billiards': 12, 'BlowDryHair': 13, 'BlowingCandles': 14, 'BodyWeightSquats': 15, 'Bowling': 16, 'BoxingPunchingBag': 17, 'BoxingSpeedBag': 18, 'BreastStroke': 19, 'BrushingTeeth': 20, 'CleanAndJerk': 21, 'CliffDiving': 22, 'CricketBowling': 23, 'CricketShot': 24, 'CuttingInKitchen': 25, 'Diving': 26, 'Drumming': 27, 'Fencing': 28, 'FieldHockeyPenalty': 29, 'FloorGymnastics': 30, 'FrisbeeCatch': 31, 'FrontCrawl': 32, 'GolfSwing': 33, 'Haircut': 34, 'Hammering': 35, 'HammerThrow': 36, 'HandstandPushups': 37, 'HandstandWalking': 38, 'HeadMassage': 39, 'HighJump': 40, 'HorseRace': 41, 'HorseRiding': 42, 'HulaHoop': 43, 'IceDancing': 44, 'JavelinThrow': 45, 'JugglingBalls': 46, 'JumpingJack': 47, 'JumpRope': 48, 'Kayaking': 49, 'Knitting': 50, 'LongJump': 51, 'Lunges': 52, 'MilitaryParade': 53, 'Mixing': 54, 'MoppingFloor': 55, 'Nunchucks': 56, 'ParallelBars': 57, 'PizzaTossing': 58, 'PlayingCello': 59, 'PlayingDaf': 60, 'PlayingDhol': 61, 'PlayingFlute': 62, 'PlayingGuitar': 63, 'PlayingPiano': 64, 'PlayingSitar': 65, 'PlayingTabla': 66, 'PlayingViolin': 67, 'PoleVault': 68, 'PommelHorse': 69, 'PullUps': 70, 'Punch': 71, 'PushUps': 72, 'Rafting': 73, 'RockClimbingIndoor': 74, 'RopeClimbing': 75, 'Rowing': 76, 'SalsaSpin': 77, 'ShavingBeard': 78, 'Shotput': 79, 'SkateBoarding': 80, 'Skiing': 81, 'Skijet': 82, 'SkyDiving': 83, 'SoccerJuggling': 84, 'SoccerPenalty': 85, 'StillRings': 86, 'SumoWrestling': 87, 'Surfing': 88, 'Swing': 89, 'TableTennisShot': 90, 'TaiChi': 91, 'TennisSwing': 92, 'ThrowDiscus': 93, 'TrampolineJumping': 94, 'Typing': 95, 'UnevenBars': 96, 'VolleyballSpiking': 97, 'WalkingWithDog': 98, 'WallPushups': 99, 'WritingOnBoard': 100, 'YoYo': 101}
#predictions = model(predict_dataset)


#for i, logits in enumerate(predictions):
#  class_idx = tf.argmax(logits).numpy()
#  p = tf.nn.softmax(logits)[class_idx]
#  name = dict(class_idx)
#  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

print('done')
i =0 
hist_frames=[]
for x,y in datagen():
    i = i+1
    if(i == 5000): break
    print(i)
    eval = model.evaluate(x,y,batch_size=16)
    hist_frames.append(eval)

print(hist_frames)

  
          
loss_list = [item[0] for item in hist_frames]
#mae_list = [item[1] for item in hist_frames]
accuracy_list= [item[2] for item in hist_frames]

#print(loss_list)
print("Average loss: ", np.average(loss_list))

#print(mae_list)
#print("Average mae: ", np.average(mae_list))

#print(accuracy_list)
print("Average accuracy: ", np.average(accuracy_list))

plt.plot(accuracy_list[0::25])
#plt.plot(mae_list[0::200])
plt.title('model accuracy')
plt.ylabel('')
plt.xlabel('')
plt.legend(['accuracy'], loc='upper left')
#plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_test_ED.png')
#plt.show()
# summarize history for loss
plt.clf()  
plt.plot(loss_list[0::25])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('')
#plt.show()
plt.savefig('loss_test_ED.png')
#result = {} 
#for d in hist_frames: 
#    for k in d: 
#        result[k] = np.add(result[k], d[k])
 
#print("resultant dictionary : ", str(result)) 
  
