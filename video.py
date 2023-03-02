import cv2 
import numpy as np
import tensorflow as tf
import keras
from keras import layers

path = "haarcascade_frontalface_default.xml"

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

rectangle_bgr = (255, 255, 255)
#make a black image
img = np.zeros((500, 500))
#set some text
text = "Some text in a box"
#get width and height of the textbox
(text_width, text_height) = cv2.getTextSize(text, font, fontScale= font_scale, thickness=1)[0]
#set text start position
text_offset_x=10
text_offset_y=img.shape[0] - 25
#make the coords of the box with a small padding of 2 pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width+2, text_offset_y- text_height -2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)

cap = cv2.VideoCapture(1)
#check if webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam!")

while True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'harcascade_frontalface_default.xml')
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(gray,1,1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]
        cv2.rectangle(frame, (x,y) , (x+w , y+h), (255, 0 , 0) , 2)
        faces = faceCascade.detectMultiScale(roi_gray)
        if len(faces)==0:
            print("Face not detected!!")
        else:
            for (ex, ey, ew , eh) in faces:
                face_roi = roi_color[ey: ey+eh , ex:ex+ew]

    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis =0) #need fourth dimension
    final_image = final_image/255.0 #normalizing

    font = cv2.FONT_HERSHEY_SIMPLEX

    model = tf.keras.applications.MobileNetV2()
    base_input = model.layers[0].input #input
    base_output = model.layers[-2].output

    final_output = layers.Dense(128)(base_output) #adding new layer after the output of global pooling layer
    final_output = layers.Activation('relu')(final_output) #activation function
    final_output = layers.Dense(64)(final_output)
    final_output = layers.Activation('relu')(final_output)
    final_output = layers.Dense(7, activation = 'softmax')(final_output) #my classes are 7

    print(final_output)

    new_model = keras.Model(inputs = base_input, outputs = final_output)

    Predictions = new_model.predict(final_image)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    if (np.argmax(Predictions) == 'angry'):
        status = "Angry"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))  

    elif (np.argmax(Predictions) == 'disgust'):
        status = "Disgust"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))

    elif (np.argmax(Predictions) == 'fear'):
        status = "Fear"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))

    elif (np.argmax(Predictions) == 'happy'):
        status = "Happy"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))
        
    elif (np.argmax(Predictions) == 'sad'):
        status = "Sad"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))

    elif (np.argmax(Predictions) == 'surprise'):
        status = "Surprise"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))

    elif (np.argmax(Predictions) == 'neutral'):
        status = "Neutral"

        x1,y1,w1,h1 = 0,0,175,75

        cv2.rectangle(frame,(x1,x1), (x1+w1 , y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) , 2)

        cv2.putText(frame, status, (100, 150) , font, 3, (0,0,255), cv2.LINE_4)

        cv2.rectangle(frame, (x,y) , (x+w, y+w), (0,0,255))

    
    cv2.imshow('Face Emotion Recognition')

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        