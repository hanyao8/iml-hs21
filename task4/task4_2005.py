#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.models import Model

from keras.applications import MobileNetV2
from keras.applications import VGG16

import pandas as pd
from datetime import datetime
import os

print(tf.__version__)
print(keras.__version__)

now = datetime.now()
current_time = now.strftime("%m_%d_%H_%M_%S")
print("Current Time =", current_time)

#setup logging
if not(os.path.exists("jobs")):
    os.mkdir("jobs")
current_job_path = os.path.join("jobs","job_"+current_time)
os.mkdir(current_job_path)

job_log = open(os.path.join(current_job_path,"log.txt"),"a")
job_log.write("init\n")

model1 = MobileNetV2(weights="imagenet")
model2 = Model(inputs=model1.inputs, outputs=model1.layers[-3].output)

print(model1.summary())
print(model2.summary())


sample = pd.read_table("data/sample.txt",header=None)
train_triplets = pd.read_table("data/train_triplets.txt",delimiter=" ",header=None,dtype=str)
test_triplets = pd.read_table("data/test_triplets.txt",delimiter=" ",header=None,dtype=str)


def image_pred(image_code,model):
    image_path = 'data/food/'+image_code+'.jpg'
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    print("yhat:")
    print(np.shape(yhat))
    #yhat = yhat[0]

    return(yhat)


def triplet_pred(triplet,model):
    triplet_feats_shape = tuple([3]+list((model.layers[-1].output_shape)[1:]))
    print(triplet_feats_shape)
    triplet_feats = np.zeros(triplet_feats_shape)
    for i in range(len(triplet)):
        triplet_feats[i] = image_pred(triplet[i],model)

    dot_01 = np.dot(triplet_feats[0].flatten(),triplet_feats[1].flatten())
    dot_02 = np.dot(triplet_feats[0].flatten(),triplet_feats[2].flatten())
    if dot_01 > dot_02:
        return(1)
    else:
        return(0)

n = train_triplets.shape[0]
y = np.zeros(n)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

for i in range(100):
    triplet = train_triplets.loc[i].values
    y[i] = triplet_pred(triplet,model2)
    print(i)
    print("\n")
    
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

job_log.close()
