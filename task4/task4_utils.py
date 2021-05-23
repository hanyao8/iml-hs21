import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

def code_2_path(image_code):
    image_path = os.path.join(*[CWD,"data","food",str(image_code)+".jpg"])
    return (image_path)

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224,224))
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])


###

def train_val_dataset_from_df(train_triplets,train_dataset_size):
    shuffled_idx = np.arange(train_triplets.shape[0])
    np.random.shuffle(shuffled_idx)

    image_count = train_dataset_size
    print(image_count)

    anchor_image_paths = []
    positive_image_paths = []
    negative_image_paths = []
    for i in range(train_dataset_size):
        anchor_code = train_triplets.iloc[shuffled_idx[i]][0]
        positive_code = train_triplets.iloc[shuffled_idx[i]][1]
        negative_code = train_triplets.iloc[shuffled_idx[i]][2]
        anchor_image_paths.append(code_2_path( anchor_code ))
        positive_image_paths.append(code_2_path( positive_code ))
        negative_image_paths.append(code_2_path( negative_code ))

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_image_paths)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_image_paths)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_image_paths)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)

    return (train_dataset,val_dataset)



def hold_triplets_from_pos(pos_hold_triplets):
    neg_hold_triplets = pos_hold_triplets.copy()
    neg_hold_triplets[1] = pos_hold_triplets[2]
    neg_hold_triplets[2] = pos_hold_triplets[1]
    hold_triplets = pd.concat([pos_hold_triplets,neg_hold_triplets])
    return (hold_triplets)

def hold_gt_from_df(hold_triplets):
    y_hold_groundtruth = np.zeros(hold_triplets.shape[0])
    y_hold_groundtruth[:int(hold_triplets.shape[0]/2)] = 1
    y_hold_groundtruth = y_hold_groundtruth.astype(int)
    return (y_hold_groundtruth)

def hold_dataset_from_df(hold_triplets):
    hold_dataset_size = hold_triplets.shape[0]

    image_count = hold_dataset_size
    print(image_count)

    anchor_image_paths = []
    positive_image_paths = []
    negative_image_paths = []
    for i in range(hold_dataset_size):
        anchor_code = hold_triplets.iloc[i][0]
        positive_code = hold_triplets.iloc[i][1]
        negative_code = hold_triplets.iloc[i][2]
        anchor_image_paths.append(code_2_path( anchor_code ))
        positive_image_paths.append(code_2_path( positive_code ))
        negative_image_paths.append(code_2_path( negative_code ))

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_image_paths)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_image_paths)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_image_paths)

    hold_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    hold_dataset = hold_dataset.map(preprocess_triplets)

    hold_dataset = hold_dataset.batch(32, drop_remainder=False)
    hold_dataset = hold_dataset.prefetch(8)

    return (hold_dataset)


def test_dataset_from_df(test_triplets):
    test_dataset_size = test_triplets.shape[0]
    #test_dataset_size = 1000
    image_count = test_dataset_size
    print(image_count)

    anchor_image_paths = []
    positive_image_paths = []
    negative_image_paths = []
    for i in range(test_dataset_size):
        anchor_code = test_triplets.iloc[i][0]
        positive_code = test_triplets.iloc[i][1]
        negative_code = test_triplets.iloc[i][2]
        anchor_image_paths.append(code_2_path( anchor_code ))
        positive_image_paths.append(code_2_path( positive_code ))
        negative_image_paths.append(code_2_path( negative_code ))

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_image_paths)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_image_paths)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_image_paths)

    test_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    test_dataset = test_dataset.map(preprocess_triplets)

    test_dataset = test_dataset.batch(32, drop_remainder=False)
    test_dataset = test_dataset.prefetch(8)

    return (test_dataset)

