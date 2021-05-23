import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from keras.models import Model

from keras.applications import MobileNetV2
from keras.applications import mobilenet

import custom_objects

###Siamese Network approach

def create_siamese():
    feature_cnn = MobileNetV2(weights="imagenet",
                              input_shape=(224,224,3),
                              include_top=False)
    
    for layer in feature_cnn.layers:
        layer.trainable = True

    flatten = layers.Flatten()(feature_cnn.output)
    dense = layers.Dense(512,activation="relu")(flatten)
    dense = layers.BatchNormalization()(dense)
    output = layers.Dense(256)(dense)
    embedding = Model(feature_cnn.input,output,name="Embedding")

    anchor_input = layers.Input(name="anchor", shape=(224,224,3))
    positive_input = layers.Input(name="positive", shape=(224,224,3))
    negative_input = layers.Input(name="negative", shape=(224,224,3))

    distances = custom_objects.DistanceLayer()(
        embedding(mobilenet.preprocess_input(anchor_input)),
        embedding(mobilenet.preprocess_input(positive_input)),
        embedding(mobilenet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return (siamese_network)

def create_siamese2():
    feature_cnn = MobileNetV2(weights="imagenet",
                              input_shape=(224,224,3),
                              include_top=False)
    
    for layer in feature_cnn.layers:
        layer.trainable = False

    flatten = layers.Flatten()(feature_cnn.output)
    dense1 = layers.Dense(512,activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256,activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)
    embedding = Model(feature_cnn.input,output,name="Embedding")

    anchor_input = layers.Input(name="anchor", shape=(224,224,3))
    positive_input = layers.Input(name="positive", shape=(224,224,3))
    negative_input = layers.Input(name="negative", shape=(224,224,3))

    distances = custom_objects.DistanceLayer()(
        embedding(mobilenet.preprocess_input(anchor_input)),
        embedding(mobilenet.preprocess_input(positive_input)),
        embedding(mobilenet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return (siamese_network)



