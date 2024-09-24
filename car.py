import cv2
import numpy as np
import os

def get_images_from_video(video_name, time_F):
    video_images = []
    vc = cv2.VideoCapture(video_name)
    c = 1
    if vc.isOpened(): #判斷是否開啟影片
        rval, video_frame = vc.read()
    else:
        rval = False
    while rval:   #擷取視頻至結束
        rval, video_frame = vc.read()

        if(c % time_F == 0): #每隔幾幀進行擷取
            video_images.append(video_frame)
        c = c + 1
    vc.release()
    return video_images

time_F = 10 #time_F越小，取樣張數越多
video_name = os. getcwd() + "/source/1.mp4" #影片名稱
video_images = get_images_from_video(video_name, time_F) #讀取影片並轉成圖片
car_cascade = cv2.CascadeClassifier(os. getcwd()+"/source/cars.xml")

def car_detect(img):
    cars_detect = car_cascade.detectMultiScale(img, 1.1, 1, minSize=(100,100),)
    for (x,y,w,h) in cars_detect:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def car_count(img):
    count = 0
    car = car_cascade.detectMultiScale(img, 1.1, 2)
    for (x,y,w,h) in car:
        count+=1
    return count

result = []
for i in range(len(video_images)):
    result.append(car_detect(video_images[i]))
''' 
for i in range(0, len(video_images)-1): #顯示出所有擷取之圖片
    cv2.imshow('windows', result[i])
    cv2.waitKey(300)

cv2.destroyAllWindows()
'''
count = 0
for i in range(len(video_images)-1):
    count+=car_count(video_images[i])
count = count/(len(video_images)-1)