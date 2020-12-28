import os
import glob as glob

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#######################################

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

N_CLASSES = 21

#######################################

def parse_record(record):
    name_to_features = {
        'rows': tf.io.FixedLenFeature([], tf.int64),
        'cols': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

def decode_record(record):
    image = tf.io.decode_raw(
        record['image'], out_type='uint8', little_endian=True, fixed_length=None, name=None
    )
    target = tf.io.decode_raw(
        record['target'], out_type='uint8', little_endian=True, fixed_length=None, name=None
    )

    rows = record['rows']
    cols = record['cols']
    depth = record['depth']

    image = tf.reshape(image, (rows,cols,depth))
    target = tf.reshape(target, (rows,cols))

    return (image,target)

#######################################

AUTO = tf.data.experimental.AUTOTUNE

def read_tf_image_and_label(record):
    parsed_record = parse_record(record)
    X, y = decode_record(parsed_record)
    X = tf.cast(X, tf.float32) / 255.0
    return X, y

def get_training_dataset(record_files):
    dataset = tf.data.TFRecordDataset(record_files, buffer_size=100)
    dataset = dataset.map(read_tf_image_and_label, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(AUTO)
    return dataset

#######################################

train_dataset = get_training_dataset("tfData/train_record.tfrecords")
train_dataset = train_dataset.shuffle(10000).batch(32)

valid_dataset = get_training_dataset("tfData/val_record.tfrecords")
valid_dataset = valid_dataset.shuffle(10000).batch(32)

#######################################

class ResNetFCN(tf.keras.Model):
    def __init__(self,input_shape=(320, 224, 3), **kwargs):
        super(ResNetFCN, self).__init__(**kwargs)
        self.model = self.getModel(input_shape)

    def getModel(self,input_shape):
        base_model = tf.keras.applications.ResNet101V2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
                input_shape=input_shape,
                include_top=False,
        )
        #base_model.trainable = False

        i = 0
        for layer in base_model.layers:
            print(layer.name)
            if i < 400:
                layer.trainable = False
            i += 1

        feat_ex = tf.keras.Model(base_model.input,base_model.layers[-10].output)

        # Build FCN
        inputs = tf.keras.Input(input_shape)
        x = feat_ex(inputs)
        x = layers.UpSampling2D(2)(x)
        for filters in [500, 400, 300, 200]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 4,kernel_regularizer= regularizers.l2(0.01), padding="same")(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(.3)(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 4,kernel_regularizer= regularizers.l2(0.01), padding="same")(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(.3)(x)

            x = layers.UpSampling2D(2)(x)

        output = layers.Conv2D(N_CLASSES, 1,kernel_regularizer= regularizers.l2(0.001),activation="softmax",padding="same")(x)
        model = tf.keras.Model(inputs,output)

        print(model.summary())
        return model

    def train_step(self, data):
        X, y = data[0], data[1]
        with tf.GradientTape() as tape:
            yh = self.model(X)
            total_loss = self.compiled_loss(y,yh)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y, yh)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y = data[0], data[1]
        yh = self.model(X)
        total_loss = self.compiled_loss(y,yh)
        self.compiled_metrics.update_state(y, yh)
        return {m.name: m.result() for m in self.metrics}

    def call(self, X):
        yh = self.model(X)
        return yh

#######################################

model = ResNetFCN()
opt = tf.keras.optimizers.Nadam(learning_rate = 0.0005)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#######################################

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="ResNetFCN_VOC2012.h5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

epochs = 30
model.fit(train_dataset,epochs=epochs,validation_data=valid_dataset,
          callbacks=model_checkpoint_callback)

model.save_weights("final_weights.h5")
