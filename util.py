from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
import numpy as np
import os


gan_path="genrator_save/genrator_weights"
AE_path = "AE_save/cp.ckpt"
noise_dim = 20
latent_dim = 64
init_w=keras.initializers.RandomNormal(mean=0,stddev=0.05)


def load_dataset():
    """
    load data set
    :return: the data set splited to train and test
    """
    # load train and test images, and pad & reshape them to (-1,32,32,1)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
    x_train = np.pad(x_train, ((0,0),(2, 2), (2, 2),(0,0)),mode='constant')
    x_test = np.pad(x_test, ((0,0),(2, 2), (2, 2),(0,0)),mode='constant')
    print(x_train.shape)
    print(x_test.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
    y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
    return x_train,y_train,x_test,y_test



def build_encoder():
    """
    build the encoder arch
    :return: the built model
    """
    encoder = Sequential()
    encoder.add(layers.Conv2D(16, (4, 4), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,1)))
    encoder.add(layers.Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
    encoder.add(layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
    encoder.add(layers.Conv2D(96, (3, 3), strides=(2,2), activation='relu', padding='same'))
    encoder.add(layers.Reshape((2*2*96,)))
    encoder.add(layers.Dense(latent_dim))
    return encoder

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
def build_decoder():
    """
        build the decoder arch
        :return: the built model
    """
    decoder = Sequential()
    decoder.add(layers.Dense(2*2*96,activation='relu', input_shape=(latent_dim,)))
    decoder.add(layers.Reshape((2,2,96)))
    decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
    decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2,2), activation='relu', padding='same'))
    decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2,2), activation='sigmoid', padding='same'))
    return decoder


def build_genrator(out_dim=64):
    """
    build the genrator arch
    :param out_dim: the dim of the latent vector
    :return: the model
    """
    genrator = Sequential()
    genrator.add(layers.Dense(25,kernel_initializer=init_w))
    genrator.add(layers.LeakyReLU(alpha=0.15))
    genrator.add(layers.Dropout(0.25))
    genrator.add(layers.Dense(30))
    genrator.add(layers.LeakyReLU(alpha=0.15))
    genrator.add(layers.Dropout(0.25))
    genrator.add(layers.Dense(35))
    genrator.add(layers.LeakyReLU(alpha=0.15))
    genrator.add(layers.Dropout(0.25))
    genrator.add(layers.Dense(40))
    genrator.add(layers.LeakyReLU(alpha=0.15))
    genrator.add(layers.Dropout(0.25))
    genrator.add(layers.Dense(50))
    genrator.add(layers.LeakyReLU(alpha=0.15))
    genrator.add(layers.Dropout(0.45))
    genrator.add(layers.Dense(out_dim))
    return genrator

def build_Cgenrator(out_dim=64):
    """
        build the CGAN arch
        :param out_dim: the dim of the latent vector
        :return: the model
    """
    in_noise=keras.Input(shape=(noise_dim,))
    label_encoding=keras.Input(shape=(10,))
    merged=layers.concatenate([in_noise,label_encoding])
    x=layers.Dense(30)(merged)
    x=layers.LeakyReLU(alpha=0.15)(x)
    x=layers.Dropout(0.25)(x)
    x = layers.Dense(35)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(40)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(45)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(55)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.45)(x)
    out=layers.Dense(out_dim)(x)
    model=keras.Model(inputs=[in_noise,label_encoding],outputs=out)
    return model

def build_discriminator():
    """
        build the discriminator arch
        :return: the model
    """
    discriminator = Sequential()
    discriminator.add(layers.Dense(160,activation='relu'))
    discriminator.add(layers.Dropout(rate=0.2))
    discriminator.add(layers.Dense(60,activation='relu'))
    discriminator.add(layers.Dropout(rate=0.2))
    discriminator.add(layers.Dense(32, activation='relu'))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    return discriminator

def build_Cdiscriminator():
    """
           build the discriminator arch
           :return: the model
    """
    label_encodeing=keras.Input(shape=(10,))
    latent_vector=keras.Input(shape=(latent_dim,))
    merged=layers.concatenate([latent_vector,label_encodeing])
    x=layers.Dense(165,activation='relu')(merged)
    x=layers.Dropout(rate=0.3)(x)
    x=layers.Dense(60,activation='relu')(x)
    x=layers.Dropout(rate=0.3)(x)
    x=layers.Dense(36,activation='relu')(x)
    out=layers.Dense(1,activation='sigmoid')(x)
    model=keras.Model(inputs=[latent_vector,label_encodeing],outputs=out)
    return model

