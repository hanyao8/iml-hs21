import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import preprocessor

def code_2_path(image_code):
    image_path = os.path.join(*["data","food",str(image_code)+".jpg"])
    return (image_path)


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

def train_val_dataset_from_df(train_triplets,y_train_groundtruth=[],
        train_dataset_size=0,val_frac=0.2,prep=None):
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

    if prep.multitask:
        y_train_gt_shuffled = np.zeros(y_train_groundtruth.shape[0])
        for i in range(train_dataset_size):
            y_train_gt_shuffled[i] = y_train_groundtruth[shuffled_idx[i]]
        y_dataset = tf.data.Dataset.from_tensor_slices(y_train_gt_shuffled)
        dataset = tf.data.Dataset.zip((dataset,y_dataset))

    #dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.shuffle(buffer_size=train_dataset_size)

    if prep.multitask:
        dataset = dataset.map(prep.preprocess_triplets_mt)
    else:
        dataset = dataset.map(prep.preprocess_triplets)

    if val_frac > 1.0e-6:
        # Let's now split our dataset in train and validation.
        train_dataset = dataset.take(round(image_count * 0.8))
        val_dataset = dataset.skip(round(image_count * 0.8))
    
        train_dataset = train_dataset.batch(prep.batch_size, drop_remainder=False)
        train_dataset = train_dataset.prefetch(8)
    
        val_dataset = val_dataset.batch(prep.batch_size, drop_remainder=False)
        val_dataset = val_dataset.prefetch(8)
    
        return (train_dataset,val_dataset)

    else:
        dataset = dataset.batch(prep.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(8)
        return (dataset)


def triplets_from_pos(pos_triplets):
    neg_triplets = pos_triplets.copy()
    neg_triplets[1] = pos_triplets[2]
    neg_triplets[2] = pos_triplets[1]
    triplets = pd.concat([pos_triplets,neg_triplets])
    return (triplets)

def gt_from_df(triplets):
    y_groundtruth = np.zeros(triplets.shape[0])
    y_groundtruth[:int(triplets.shape[0]/2)] = 1
    y_groundtruth = y_groundtruth.astype(int)
    return (y_groundtruth)

def hold_dataset_from_df(hold_triplets,prep):
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

    hold_dataset = hold_dataset.map(prep.preprocess_triplets)

    hold_dataset = hold_dataset.batch(prep.batch_size, drop_remainder=False)
    hold_dataset = hold_dataset.prefetch(8)

    return (hold_dataset)


def test_dataset_from_df(test_triplets,prep):
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

    test_dataset = test_dataset.map(prep.preprocess_triplets)

    test_dataset = test_dataset.batch(prep.batch_size, drop_remainder=False)
    test_dataset = test_dataset.prefetch(8)

    return (test_dataset)


def get_d(model,dataset,dataset_size,prep):
    batch_size = prep.batch_size
    n_full_batches = int(dataset_size/batch_size)
    d = np.zeros( ( dataset_size,2 ) )

    now = datetime.now()
    this_time = now.strftime("%m_%d_%H_%M_%S")
    print("This Time =", this_time)

    iterator = iter(dataset)
    for i in range(n_full_batches):
        sample = iterator.get_next()
        pred = model.predict_on_batch(sample)
        d[i*batch_size:(i+1)*batch_size,0] = pred[0].flatten()
        d[i*batch_size:(i+1)*batch_size,1] = pred[1].flatten()
        if i%200==0:
            print(i)

    now = datetime.now()
    this_time = now.strftime("%m_%d_%H_%M_%S")
    print("This Time =", this_time)

    print("\n\n\n")

    if dataset_size > n_full_batches*batch_size:
        #last batch (size=24)
        sample = iterator.get_next()
        pred = model.predict_on_batch(sample)

        print(d)
        print(d[-40:])
        final_start = n_full_batches*batch_size
        final_batch_size = sample[0].shape[0]
        print(final_batch_size)
        d[final_start:final_start+final_batch_size,0] = pred[0].flatten()
        d[final_start:final_start+final_batch_size,1] = pred[1].flatten()
        print(d)
        print(d[-40:])

    return (d)





