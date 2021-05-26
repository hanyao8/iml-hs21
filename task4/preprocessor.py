import tensorflow as tf
import os

class Preprocessor():
    def __init__(self,
            target_shape,batch_size,multitask):
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.multitask = multitask

    def preprocess_image(self,filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
    
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.target_shape)
        return image

    def preprocess_triplets(self,anchor, positive, negative):
        return (
            self.preprocess_image(anchor),
            self.preprocess_image(positive),
            self.preprocess_image(negative),
        )

    def preprocess_triplets_mt(self,x,y):
        a,p,n = self.preprocess_triplets(x[0],x[1].x[2])
        return ((a,p,n),y)


