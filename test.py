import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

img_array = cv2.imread("data/test/happy/PrivateTest_218533.jpg")
print(img_array.shape) #rgb

print(plt.imshow(img_array)) #BGR

Datadirectory = "data/train"  #training dataset
Classes= ["angry" , "disgust" , "fear" , "happy" , "neutral", "sad", "surprise"] #list of classes => exact name of your folders

for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break
#resizing the image.....
img_size = 224 #ImageNet => 224 x 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()

print(new_array.shape)

#read all images and convert them to array
training_Data = [] #data array

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_Data()

print(len(training_Data))

temp = np.array(training_Data)
print(temp.shape)

import random
random.shuffle(training_Data)

x = [] #data/feature
y = [] # label

for features, label in training_Data:
    x.append(features)
    y.append(label)

x= np.array(x).reshape(-1 , img_size, img_size, 3) #converting in to 4 dimension
print(x.shape)

#normalise the data
x = x/255.0 

y[0]
y= np.array(y)
print(y.shape)

#deep learning model for training - Transfer learning
import tensorflow
from tensorflow import keras
from keras import layers

model = tf.keras.applications.MobileNetV2() #Pre trained model
print(model.summary())

#transfer learning - tuning, weights will start from last check point
base_input =model.layers[0].input
base_output = model.layers[-2].output
print(base_output)

#Transfer learning- Tuning, weights will satrt from last check point

base_input = model.layers[0].input #input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) #adding new layer after the output of global pooling layer
final_output = layers.Activation('relu')(final_output) #activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation = 'softmax')(final_output) #my classes are 7

print(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)
print(new_model.summary())

new_model.compile(loss= "sparse_categorical_crossentrophy", optimizer = "adam", metrics= ["accuracy"])
new_model.fit(x,y, epochs = 25) #training
new_model.save('Final_model_95pa07.h5')
new_model = tf.keras.models.load_model('Final_model_95pa07.h5')
frame = cv2.imread("img1.jpeg")
print(frame.shape)

#download https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# face dtection algorithm is needed (gray image)
faceCascade = cv2.CascaseClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_defalut.xml')
gray = cv2.cvtColor(frame, cv2.COLOR.BGR2GRAY)
gray.shape

faces = faceCascade.detectMultiScale(gray,1,1,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h , x:x+w]
    roi_color = frame[y:y+h , x:x+w]
    cv2.rectangle(frame, (x,y) , (x+w , y+h), (0, 255 , 0) , 2)
    faces = faceCascade.detectMultiScale(roi_gray)
    if len(faces)==0:
        print("Face not detected!!")
    else:
        for (ex, ey, ew , eh) in faces:
            face_roi = roi_color[ey: ey+eh , ex:ex+ew]

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

final_image = cv2.resize(face_roi, (224, 224))
final_image = np.expand_dims(final_image, axis =0) #need fourth dimension
final_image = final_image/255.0 #normalizing

Predictions = new_model.predict(final_image)
print(Predictions[0])

np.argmax(Predictions)