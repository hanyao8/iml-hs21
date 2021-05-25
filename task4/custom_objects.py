import tensorflow as tf
  
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from keras.models import Model

###Siamese Network approach

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class NDotLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, anchor, positive, negative):
        ap_ndot = tf.reduce_sum(tf.multiply(anchor,positive),-1)
        an_ndot = tf.reduce_sum(tf.multiply(anchor,negative),-1)
        return (ap_ndot, an_ndot)

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



class SiameseModel2(Model):
    """
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel2, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.pos_frac_tracker = metrics.Mean(name="pos_frac")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        pos_frac = self._compute_pos_frac(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        self.pos_frac_tracker.update_state(pos_frac)
        return {"loss": self.loss_tracker.result(),
                "pos_frac": self.pos_frac_tracker.result()
                }

    def test_step(self, data):
        loss = self._compute_loss(data)
        pos_frac = self._compute_pos_frac(data)
        self.loss_tracker.update_state(loss)
        self.pos_frac_tracker.update_state(pos_frac)
        return {"loss": self.loss_tracker.result(),
                "pos_frac": self.pos_frac_tracker.result()
                }

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def _compute_pos_frac(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        pos_frac = an_distance - ap_distance
        pos_frac = tf.math.sign(pos_frac)
        pos_frac = tf.maximum(pos_frac,0.0)
        return pos_frac

    @property
    def metrics(self):
        return [self.loss_tracker,self.pos_frac_tracker]
