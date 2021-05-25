import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from keras.models import Model

from keras.applications import MobileNetV2
from keras.applications import mobilenet

from keras.applications import Xception
from keras.applications import xception

import custom_objects


###Siamese Network approach

MOBILENET_INPUT_SHAPE = (224,224,3)
XCEPTION_INPUT_SHAPE = (299,299,3)

def create_siamese():
    feature_cnn = MobileNetV2(weights="imagenet",
                              input_shape=MOBILENET_INPUT_SHAPE,
                              include_top=False)
    
    for layer in feature_cnn.layers:
        layer.trainable = True

    flatten = layers.Flatten()(feature_cnn.output)
    dense = layers.Dense(512,activation="relu")(flatten)
    dense = layers.BatchNormalization()(dense)
    output = layers.Dense(256)(dense)
    embedding = Model(feature_cnn.input,output,name="Embedding")

    anchor_input = layers.Input(name="anchor", shape=MOBILENET_INPUT_SHAPE)
    positive_input = layers.Input(name="positive", shape=MOBILENET_INPUT_SHAPE)
    negative_input = layers.Input(name="negative", shape=MOBILENET_INPUT_SHAPE)

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
                              input_shape=MOBILENET_INPUT_SHAPE,
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

    anchor_input = layers.Input(name="anchor", shape=MOBILENET_INPUT_SHAPE)
    positive_input = layers.Input(name="positive", shape=MOBILENET_INPUT_SHAPE)
    negative_input = layers.Input(name="negative", shape=MOBILENET_INPUT_SHAPE)

    distances = custom_objects.DistanceLayer()(
        embedding(mobilenet.preprocess_input(anchor_input)),
        embedding(mobilenet.preprocess_input(positive_input)),
        embedding(mobilenet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return (siamese_network)


def create_siamese_xception():
    feature_cnn = Xception(weights="imagenet",
                              input_shape=XCEPTION_INPUT_SHAPE,
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

    anchor_input = layers.Input(name="anchor", shape=XCEPTION_INPUT_SHAPE)
    positive_input = layers.Input(name="positive", shape=XCEPTION_INPUT_SHAPE)
    negative_input = layers.Input(name="negative", shape=XCEPTION_INPUT_SHAPE)

    distances = custom_objects.DistanceLayer()(
        embedding(xception.preprocess_input(anchor_input)),
        embedding(xception.preprocess_input(positive_input)),
        embedding(xception.preprocess_input(negative_input)),
    )

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return (siamese_network)


def create_siamese_mobilenet_dot_2():
    feature_cnn = MobileNetV2(weights="imagenet",
                              input_shape=MOBILENET_INPUT_SHAPE,
                              include_top=True)

    trainable_layer_names = ["Conv_1","predictions"]
    for layer in feature_cnn.layers:
        if layer.name in trainable_layer_names:
            layer.trainable = True
        else:
            layer.trainable = False

    print("Trainable check:")
    for layer in feature_cnn.layers:
        print(layer.name)
        print(layer.trainable)
        print("\n")

    flatten = layers.Flatten()(feature_cnn.output)
    embedding = Model(feature_cnn.input,flatten,name="Embedding")

    anchor_input = layers.Input(name="anchor", shape=MOBILENET_INPUT_SHAPE)
    positive_input = layers.Input(name="positive", shape=MOBILENET_INPUT_SHAPE)
    negative_input = layers.Input(name="negative", shape=MOBILENET_INPUT_SHAPE)

    anchor_embedding = embedding(anchor_input)
    positive_embedding = embedding(positive_input)
    negative_embedding = embedding(negative_input)

    ap_dot = layers.Dot(axes=1)([anchor_embedding,positive_embedding])
    an_dot = layers.Dot(axes=1)([anchor_embedding,negative_embedding])

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=(1.0-ap_dot,1.0-an_dot))

    return (siamese_network)


def create_siamese_xception_dot():

    feature_cnn = Xception(weights="imagenet",
                              input_shape=XCEPTION_INPUT_SHAPE,
                              include_top=True)

    trainable_layer_names = ["predictions",
            "block14_sepconv2",
            "block14_sepconv1"]
    for layer in feature_cnn.layers:
        layer.trainable = False
        if (layer.name)[-3:]=="_bn":
            layer.trainable = True
        if layer.name in trainable_layer_names:
            layer.trainable = True

    print("Trainable check:")
    for layer in feature_cnn.layers:
        print(layer.name)
        print(layer.trainable)
        print("\n")

    flatten = layers.Flatten()(feature_cnn.output)
    embedding = Model(feature_cnn.input,flatten,name="Embedding")

    anchor_input = layers.Input(name="anchor", shape=MOBILENET_INPUT_SHAPE)
    positive_input = layers.Input(name="positive", shape=MOBILENET_INPUT_SHAPE)
    negative_input = layers.Input(name="negative", shape=MOBILENET_INPUT_SHAPE)

    anchor_embedding = embedding(anchor_input)
    positive_embedding = embedding(positive_input)
    negative_embedding = embedding(negative_input)

    ap_dot = layers.Dot(axes=1)([anchor_embedding,positive_embedding])
    an_dot = layers.Dot(axes=1)([anchor_embedding,negative_embedding])

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=(1.0-ap_dot,1.0-an_dot))

    return (siamese_network)
