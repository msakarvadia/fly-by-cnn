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
import argparse

def main(args):
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


    class BahdanauAttention(tf.keras.layers.Layer):
            def __init__(self, units):
                super(BahdanauAttention, self).__init__()
                self.W1 = tf.keras.layers.Dense(units)
                self.W2 = tf.keras.layers.Dense(units)
                self.V = tf.keras.layers.Dense(1)

            def call(self, query, values):
                # query hidden state shape == (batch_size, hidden size)
                # query_with_time_axis shape == (batch_size, 1, hidden size)
                # values shape == (batch_size, max_len, hidden size)
                # we are doing this to broadcast addition along the time axis to calculate the score
                query_with_time_axis = tf.expand_dims(query, 1)

                # score shape == (batch_size, max_length, 1)
                # we get 1 at the last axis because we are applying score to self.V
                # the shape of the tensor before applying self.V is (batch_size, max_length, units)
                score = self.V(tf.nn.tanh(
                            self.W1(query_with_time_axis) + self.W2(values)))

                # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]])
                # min_score = tf.reshape(min_score, [-1, 1, 1])
                # score_mask = tf.greater_equal(score, min_score)
                # score_mask = tf.cast(score_mask, tf.float32)
                # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multi

                # attention_weights shape == (batch_size, max_length, 1)
                attention_weights = tf.nn.softmax(score, axis=1)

                # context_vector shape after sum == (batch_size, hidden_size)
                context_vector = attention_weights * values
                context_vector = tf.reduce_sum(context_vector, axis=1)

                return context_vector, attention_weights


    def make_conv_net(drop_prob=0):
            x0 = tf.keras.Input(shape=[256, 256, args.num_features])
            #x0 = tf.keras.Input(shape=[256, 256, 6])
            #x0 = tf.keras.Input(shape=[256, 256, 4])

            x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same')(x0)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D()(x)
            x = layers.Dropout(drop_prob)(x)

            x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D()(x)
            x = layers.Dropout(drop_prob)(x)
            d0 = x

            x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D()(x)
            x = layers.Dropout(drop_prob)(x)
            d2 = x

            x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D()(x)
            x = layers.Dropout(drop_prob)(x)
            d3 = x

            x = layers.GlobalMaxPooling2D()(x)

            return tf.keras.Model(inputs=x0, outputs=x)


    def make_gru_network(drop_prob=0):
            x0 = tf.keras.Input(shape=[None, 256, 256, args.num_features])

            conv_net = make_conv_net()
            x_e = layers.TimeDistributed(conv_net)(x0)
            x_h = layers.Bidirectional(layers.GRU(units=512, activation='tanh', use_bias=False, kernel_initializer="glorot_normal", dropout=drop_prob))(x_e)
            
            x_e = layers.Dropout(drop_prob)(x_e)
            x_h = layers.Dropout(drop_prob)(x_h)

            x_a, w_a_fwd = BahdanauAttention(1024)(x_h, x_e)

            x = tf.concat([x_h, x_a], axis=-1)
            x = layers.Dense(1, activation='sigmoid', use_bias=False, name='predictions')(x)

            return tf.keras.Model(inputs=x0, outputs=x)


    class DatasetGenerator:
        def __init__(self, df):
            self.df = df

            self.dataset = tf.data.Dataset.from_generator(self.generator,
                output_types=(tf.float32, tf.int32), 
                output_shapes=((16, 256, 256, args.num_features), [1])
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

                #TODO this is the depth and the normals
                img_np = img_np[:,:,:,0:4]

                img_np = img_np.reshape([16, 256, 256, args.num_features])

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

    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    dataset = DatasetGenerator(train_df).get()
    dataset_validation = DatasetGenerator(valid_df).get()
    dataset_test = DatasetGenerator(test_df).get()


# with strategy.scope():

    batch_size = 4

    model = make_gru_network()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

    #checkpoint_path = "/work/mansisak/trained_models/gru_cnn_nn_large_set_sa_ct_sd_depth_norm"
    checkpoint_path = args.model_checkpoint_dir


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)
    model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

    model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])

    model.load_weights(checkpoint_path)
    loss, acc = model.evaluate(dataset_test, verbose=2)
    print("Trained model, accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model to predict labels using FlyBy image sequences')
    parser.add_argument('--train_csv', type=str, default=None, help='CSV of training set')
    parser.add_argument('--test_csv', type=str, default=None, help='CSV of test set')
    parser.add_argument('--val_csv', type=str, default=None, help='CSV of validation set')
    parser.add_argument('--model_checkpoint_dir', type=str, default="model", help='Path to directory where model should be saved')
    parser.add_argument('--num_features', type=int, default=3, help='Number of features displayed on surface')


args = parser.parse_args()


main(args)
