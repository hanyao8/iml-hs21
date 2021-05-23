import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from keras.models import Model

from keras.applications import MobileNetV2
from keras.applications import mobilenet

###Siamese Network approach

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

def create_siamese1():
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

    distances = DistanceLayer()(
        embedding(mobilenet.preprocess_input(anchor_input)),
        embedding(mobilenet.preprocess_input(positive_input)),
        embedding(mobilenet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return (siamese_network)


class SiameseModel(Model):
    """
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

