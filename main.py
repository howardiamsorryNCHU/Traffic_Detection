import sys
import os
sys.path.append(os.getcwd())
import car
import walkman
import cv2

traffic_quantity = car.count
pedestrian_quantity = walkman.count*3
proportion = traffic_quantity / pedestrian_quantity 

if traffic_quantity > pedestrian_quantity:
    print('There are more cars')
else:
    print('There are more pedestrians')



for i in range(0, len(walkman.video_images)-1): #顯示出所有擷取之圖片
    cv2.imshow('Pedestrian', walkman.result[i])
    cv2.waitKey(200)
cv2.destroyAllWindows()

for i in range(0, len(car.video_images)-1): #display the frames
    cv2.imshow('Traffic', car.result[i])
    cv2.waitKey(150)
cv2.destroyAllWindows()

#%% predict traffic accident
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.decomposition import PCA

dataset_train = pd.read_csv(os.getcwd()+"/source/traffic.csv")  # 讀取訓練集 2015年11/24至2020/11/24
training_set = dataset_train.iloc[:,0:3].values

X_train = training_set[:,2:3] 
scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
y_train = training_set[:,0] 

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 3, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 6))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)


X_predict = np.array([[proportion]])
X_predict = scalar.transform(X_predict)
accidents = regressor.predict(X_predict) #predict how many accident may occur
accidents = scalar.inverse_transform(accidents) 
print("There are %s accident may happen" %accidents[0])