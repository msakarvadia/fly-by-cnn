from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
import os
import glob
import sys
import pandas as pd
import itk
from sklearn.model_selection import train_test_split

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DatasetGenerator:
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            #TODO update size
            output_shapes=((16, 256, 256, 7), [1])
            )

        self.dataset = self.dataset.batch(16)
        self.dataset = self.dataset.prefetch(48)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = row["File_path"]
            label = row["Label"]

            ImageType = itk.VectorImage[itk.F, 3]
            img_read = itk.ImageFileReader[ImageType].New(FileName=img)
            img_read.Update()
            img_np = itk.GetArrayViewFromImage(img_read.GetOutput())
            #TODO update size
            img_np = img_np.reshape([16, 256, 256, 7])

            if label==1:
                label = 0
            else:
                label = 1
            yield img_np, np.array([label])
            #yield img_np, tf.one_hot(label, 1, on_value=1.0, off_value=0.0)


gpus_index = [0]
print("Using gpus:", 0)
gpus = tf.config.list_physical_devices('GPU')


if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        gpu_visible_devices = []
        for i in gpus_index:
            gpu_visible_devices.append(gpus[i])
        
        print(bcolors.OKGREEN, "Using gpus:", gpu_visible_devices, bcolors.ENDC)

        tf.config.set_visible_devices(gpu_visible_devices, 'GPU')
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(bcolors.FAIL, e, bcolors.ENDC)

train_df = pd.read_csv("/ASD/mansi_flyby/large_set_surf_sa_ct_sd_depth_norm_DL_sets/train.csv")
valid_df = pd.read_csv("/ASD/mansi_flyby/large_set_surf_sa_ct_sd_depth_norm_DL_sets/val.csv")
test_df = pd.read_csv("/ASD/mansi_flyby/large_set_surf_sa_ct_sd_depth_norm_DL_sets/test.csv")

dataset = DatasetGenerator(train_df).get()
dataset_validation = DatasetGenerator(valid_df).get()
dataset_test = DatasetGenerator(test_df).get()

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

#TODO Change checkpoints
checkpoint_path = "/work/mansisak/trained_models/gru_cnn_nn_large_set_sa_ct_sd_depth_norm"

trained_model = tf.keras.models.load_model(checkpoint_path)

trained_model.summary()

loss, acc = trained_model.evaluate(dataset_test, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
