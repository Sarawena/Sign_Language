 
import streamlit as st
import tensorflow as tf
from threading import Thread
import time,os
import pandas as pd
import numpy as np
import cv2 as cv
import random
from PIL import Image
from os import listdir


st.title("  ِArabic Sign Language Recognition system ")
#st.button("Start")

menu=["Home", "Recognition"]
choice = st.sidebar.selectbox("Menu",menu)

#First page 
if choice == "Home": 
	st.subheader("Home page")
	
	
	st.subheader("Arabic sign language alphabets translator designed as a vision-based system using deep learning approach.")

	path=r"dataset"
	categories=[]
	for i in listdir(path):
		categories.append(i)

	typ = st.selectbox("Select the class of the finger-spelling:",np.array(categories))
	st.write("These are some of the samples from category : ",typ)
	images=[]
	for k in range(6):
		i=random.choice(listdir(os.path.join(path,typ)))

		images.append(cv.imread(os.path.join(path,typ,i)))

	st.image(Image.fromarray(np.hstack((images[0],images[1],images[2]))),width=500)
	st.image(Image.fromarray(np.hstack((images[3],images[4],images[5]))),width=500)
	
	
#Second page 
elif choice == "Recognition": 
	st.subheader("Recognition Area")
	
	model = tf.keras.models.load_model("Alexnet_v1.model")
	labels = [i for i in os.listdir('dataset/')]

	def prepare(filepath):
		IMG_SIZE = 64 
		result = cv.cvtColor(filepath , cv.COLOR_BGR2GRAY)
		new_array = cv.resize(result, (IMG_SIZE, IMG_SIZE))
		return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	frame_width = 1100
	frame_height = 800

	x = int(frame_width*0.20)
	y = 100
	w = int(frame_width*0.60)
	h = 230

	
	cap = cv.VideoCapture(0)
	frameST = st.empty()
	croppedST = st.empty()
	label = st.empty()
	stopper_started = False
	while True:
		success, frame = cap.read()
		frame = cv.resize(frame,(frame_width,frame_height))
		frame = cv.flip(frame,1)
		frame2 = frame.copy()
		cv.rectangle(frame ,(x,y),(w,y+h),(0,255,0),2)
		
		new_image = frame2[y:y+h,x:w]
		
		prediction = model.predict([prepare(new_image)])
		predicted_label = labels[np.argmax(prediction)]
		cv.putText(frame,"Put your hand in below box",(x,y-30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
		cv.putText(frame,"Predicted Sign : {}".format(predicted_label),(x-150,y+h+30),cv.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),2)
		frameST.image(frame, channels="BGR")

		label.info('**Predicted label : {}**'.format(predicted_label))

	
#elif choice == "About": 
#	st.subheader("About the project")
	



	


		
		