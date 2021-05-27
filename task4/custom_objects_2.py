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



class SiameseModel5(Model):
    """
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5, mt_weights=[]):
        super(SiameseModel5, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.lambda_1 = mt_weights[0]
        self.lambda_2 = mt_weights[1]

        self.using_labels = True
        self.bce_op = tf.keras.losses.BinaryCrossentropy()
        self.acc_op = tf.keras.metrics.Accuracy()
        self.loss_tracker = metrics.Mean(name="loss")
        self.triplet_loss_tracker = metrics.Mean(name="triplet_loss")
        self.binary_loss_tracker = metrics.Mean(name="binary_loss")
        self.acc_tracker = metrics.Mean(name="acc")

    def call(self, inputs):
        if len(inputs) == 2:
            return self.siamese_network(inputs[0])
        elif len(inputs) == 3:
            return self.siamese_network(inputs)
        elif len(inputs) == 4:
            return self.siamese_network(
                    (inputs[0],inputs[1],inputs[2])
                    )
        else:
            raise(Exception)

    def train_step(self, data):
        if self.using_labels:
            x,y = data
        else:
            x = data

        with tf.GradientTape() as tape:
            ap_distance, an_distance = self.siamese_network(x)
            d_pred = (ap_distance,an_distance)
            triplet_loss = self._compute_triplet_loss(d_pred,y)
            binary_loss = self._compute_binary_loss(d_pred,y)
            loss = self._compute_loss(triplet_loss,binary_loss)

        acc = self._compute_acc(d_pred,y)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.binary_loss_tracker.update_state(binary_loss)
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result(),
                "triplet_loss": self.triplet_loss_tracker.result(),
                "binary_loss": self.binary_loss_tracker.result()
                }

    def test_step(self, data):
        if self.using_labels:
            x,y = data
        else:
            x = data

        ap_distance, an_distance = self.siamese_network(x)
        d_pred = (ap_distance,an_distance)

        triplet_loss = self._compute_triplet_loss(d_pred,y)
        binary_loss = self._compute_binary_loss(d_pred,y)
        loss = self._compute_loss(triplet_loss,binary_loss)
        acc = self._compute_acc(d_pred,y)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.binary_loss_tracker.update_state(binary_loss)
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result(),
                "triplet_loss": self.triplet_loss_tracker.result(),
                "binary_loss": self.binary_loss_tracker.result()
                }


    def _compute_triplet_loss(self,d_pred,y_true):
        triplet_loss = ap_distance - an_distance

        y_true = tf.cast(y_true,tf.float32)
        triplet_loss = tf.math.multiply(triplet_loss,(y_true*2.0)-1.0)
        triplet_loss = tf.maximum(triplet_loss + self.margin, 0.0)
        return triplet_loss

    def _compute_triplet_loss_2(self,d_pred,y_true):
        triplet_loss = ap_distance - an_distance
        triplet_loss = tf.maximum(triplet_loss + self.margin, 0.0)

        y_true = tf.cast(y_true,tf.float32)
        y_true_sum = tf.reduce_sum(y_true)
        triplet_loss = tf.math.multiply(triplet_loss,y_true)
        triplet_loss = tf.reduce_sum(triplet_loss)/y_true_sum
        #(+) "Ceil to 1"
        return triplet_loss

    def _compute_binary_loss(self,d_pred,y_true):
        Z = tf.math.exp(-1.0*an_distance)+tf.math.exp(-1.0*ap_distance)
        phat = tf.math.exp(-1.0*ap_distance)/Z
        bce = self.bce_op(y_true,phat)
        return bce

    def _compute_loss(self, triplet_loss, binary_loss):
        loss = self.lambda_1*triplet_loss + self.lambda_2*binary_loss
        return loss

    def _compute_acc(self, d_pred, y_true):
        y_pred = an_distance - ap_distance
        y_pred = tf.math.sign(y_pred)
        y_pred = tf.maximum(y_pred,0.0)
        y_pred = tf.cast(y_pred,tf.int8)

        y_true = tf.cast(y_true,tf.int8)

        #abs_diff_sum = tf.math.reduce_sum(tf.math.abs(y_pred-y_true))
        #n_y_true = tf.cast(tf.size(tf.reshape(y_true,[-1])),tf.float32)
        #acc = (n_y_true-abs_diff_sum)/n_y_true

        self.acc_op.reset_state()
        self.acc_op.update_state(y_true,y_pred)
        acc = self.acc_op.result()
        return acc

    @property
    def metrics(self):
        return [self.loss_tracker,self.acc_tracker,
                self.triplet_loss_tracker,self.binary_loss_tracker]




class SiameseModel6(Model):
    """
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel6, self).__init__()
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





class SiameseModel7(Model):
    def __init__(self, siamese_network, margin=0.5, mt_weights=[]):
        super(SiameseModel7, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.lambda_1 = mt_weights[0]
        self.lambda_2 = mt_weights[1]

        self.using_labels = True
        self.bce_op = tf.keras.losses.BinaryCrossentropy()
        self.acc_op = tf.keras.metrics.Accuracy()
        self.loss_tracker = metrics.Mean(name="loss")
        self.triplet_loss_tracker = metrics.Mean(name="triplet_loss")
        self.binary_loss_tracker = metrics.Mean(name="binary_loss")
        self.acc_tracker = metrics.Mean(name="acc")

    def call(self, inputs):
        if len(inputs) == 2:
            return self.siamese_network(inputs[0])
        elif len(inputs) == 3:
            return self.siamese_network(inputs)
        elif len(inputs) == 4:
            return self.siamese_network(
                    (inputs[0],inputs[1],inputs[2])
                    )
        else:
            raise(Exception)

    def train_step(self, data):
        if self.using_labels:
            x,y = data
        else:
            x = data

        with tf.GradientTape() as tape:
            ap_distance, an_distance = self.siamese_network(x)
            d_pred = (ap_distance,an_distance)
            triplet_loss = self._compute_triplet_loss(d_pred,y)
            binary_loss = self._compute_binary_loss(d_pred,y)
            loss = self._compute_loss(triplet_loss,binary_loss)

        acc = self._compute_acc(d_pred,y)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.binary_loss_tracker.update_state(binary_loss)
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result(),
                "triplet_loss": self.triplet_loss_tracker.result(),
                "binary_loss": self.binary_loss_tracker.result()
                }

    def test_step(self, data):
        if self.using_labels:
            x,y = data
        else:
            x = data

        ap_distance, an_distance = self.siamese_network(x)
        d_pred = (ap_distance,an_distance)

        triplet_loss = self._compute_triplet_loss(d_pred,y)
        binary_loss = self._compute_binary_loss(d_pred,y)
        loss = self._compute_loss(triplet_loss,binary_loss)
        acc = self._compute_acc(d_pred,y)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.binary_loss_tracker.update_state(binary_loss)
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result(),
                "triplet_loss": self.triplet_loss_tracker.result(),
                "binary_loss": self.binary_loss_tracker.result()
                }


    def _compute_triplet_loss(self,d_pred,y_true):
        ap_distance,an_distance = d_pred
        triplet_loss = ap_distance - an_distance
        y_true = tf.reshape(y_true,shape=y_pred.shape)
        y_true = tf.cast(y_true,tf.float32)
        triplet_loss = tf.math.multiply(triplet_loss,(y_true*2.0)-1.0)
        triplet_loss = tf.maximum(triplet_loss + self.margin, 0.0)
        return triplet_loss

    def _compute_binary_loss(self,d_pred,y_true):
        ap_distance,an_distance = d_pred
        y_true = tf.reshape(y_true,shape=y_pred.shape)
        y_true = tf.cast(y_true,tf.float32)
        Z = tf.math.exp(-1.0*an_distance)+tf.math.exp(-1.0*ap_distance)
        phat = tf.math.exp(-1.0*ap_distance)/Z
        bce = -y_true*tf.math.log(phat)-(1-y_true)*tf.math.log(1-phat)
        return bce

    def _compute_loss(self, triplet_loss, binary_loss):
        loss = self.lambda_1*triplet_loss + self.lambda_2*binary_loss
        return loss

    def _compute_acc(self, d_pred, y_true):
        ap_distance,an_distance = d_pred
        y_pred = an_distance - ap_distance
        y_pred = tf.math.sign(y_pred)
        y_true = tf.reshape(y_true,shape=y_pred.shape)
        y_true = tf.cast(y_true,tf.float32)
        y_true = y_true*2.0-1.0
        acc = tf.multiply(y_pred,y_true)
        acc = tf.maximum(acc,0.0)
        return acc

    @property
    def metrics(self):
        return [self.loss_tracker,self.acc_tracker,
                self.triplet_loss_tracker,self.binary_loss_tracker]
