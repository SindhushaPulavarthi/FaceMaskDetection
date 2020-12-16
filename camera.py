import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

maskNet = load_model("mask_detector.model")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        mfaces=[]
        for (x,y,w,h) in face_rects:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            xface = image[y:y + h, x:x + w].copy()
            face = cv2.cvtColor(xface, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            mfaces.append(face)
            mface = np.array(mfaces, dtype="float32")
            pred = maskNet.predict(mface, batch_size=32)
            (mask, withoutMask) = pred[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            cv2.putText(image, f' {label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
