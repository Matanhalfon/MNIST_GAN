from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from util import  load_dataset,build_encoder,build_decoder,AE_path,noise_dim,build_genrator,build_discriminator,build_Cgenrator,build_Cdiscriminator
import numpy as np
import os


tf.random.set_seed(1234)
sml_train_size = 50
BATCH_SIZE = 32
Epochs=100
is_CGAN=False
load_w=False
save_w=False


def genrator_loss(fake_out):
    """
    the loss of the generator
    :param fake_out: the prediction of the discriminator over the genrated vectors
    :return: the loss by cross enroty
    """
    return (cross_entropy(tf.ones_like(fake_out), fake_out))


def discrimenator_loss(real_out, fake_out):
    """
    the loss of the discrimenator
    :param real_out: the predection over the real vectors
    :param fake_out: the predection over the fake vectors
    :return: the sumed  loss by cross enthropy
    """
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    total_loss = real_loss + fake_loss
    return total_loss



@tf.function
def train_both(latent_codec,noise):
    """
    train step for discriminator and generator by Gradients tape
    :param latent_codec: real latet vectors
    :param noise: noise vectors
    :return: applying the Gradients
    """
    with  tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        fake_vectors = genrator(noise)
        real_out = disc(latent_codec)
        fake_out = disc(fake_vectors)
        disc_loss = discrimenator_loss(real_out, fake_out)
        gen_loss = genrator_loss(fake_out)
        disc_train_loss(disc_loss)
        gen_train_loss(gen_loss)
        disc_accuracy(tf.zeros_like(fake_out), fake_out)
        gen_accuracy(tf.ones_like(fake_out), fake_out)

    gradients_of_generator = gen_tape.gradient(gen_loss, genrator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, genrator.trainable_variables))
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))


@tf.function
def train_both_c_gan(latent_codec,latent_label,noise,gen_labels):
    """
    train step for both discriminator and generator by Gradients tape,using the label
    :param latent_codec: the latent vectors
    :param latent_label:  the labels of the vectors
    :param noise: noise vectors
    :param gen_labels: labels for the noise vectors
    :return: applying the Gradients
    """
    with  tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        fake_vectors = genrator([noise,gen_labels])
        real_out = disc([latent_codec,latent_label])
        fake_out = disc([fake_vectors,gen_labels])
        disc_loss = discrimenator_loss(real_out, fake_out)
        gen_loss = genrator_loss(fake_out)
        disc_train_loss(disc_loss)
        gen_train_loss(gen_loss)
        disc_accuracy(tf.zeros_like(fake_out), fake_out)
        gen_accuracy(tf.ones_like(fake_out), fake_out)

    gradients_of_generator = gen_tape.gradient(gen_loss, genrator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, genrator.trainable_variables))
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

@tf.function
def get_noise_by_label(label):
    return tf.random.normal([noise_dim],mean=label)

def train_gan(data_set,y_labels=None):
    """
    train GAN
    :param data_set: the latent vectors to train on
    :param y_labels: if there is a y_labels will create a CGAN

    """

    for epoch in range(Epochs):
        disc_train_loss.reset_states()
        gen_train_loss.reset_states()
        disc_accuracy.reset_states()
        gen_accuracy.reset_states()
        for i,batch in enumerate(data_set):
            noise = tf.random.normal([BATCH_SIZE, noise_dim], seed=8)

            if y_labels is not None:
                labels = tf.random.uniform(shape=[BATCH_SIZE], minval=0, maxval=9, dtype=tf.int32, seed=10)
                gen_labels=tf.one_hot(labels,depth=10)
                batch_labels=y_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                train_both_c_gan(batch,batch_labels,noise,gen_labels)
            else:
                train_both(batch,noise)

        if epoch%5==0:
            plot_figs("gan at epoch "+str(epoch),data_set,labels=y_labels)
        print(
            f'Epoch {epoch + 1}, '
            f'Loss_gen: {gen_train_loss.result()}, '
            f'Loss_disc: {disc_train_loss.result()}, '
            f'disc_acc: {disc_accuracy.result()}, '
            f'gen_acc: {gen_accuracy.result()}, '
        )


def plot_figs(title,data_set,labels=None):
    """
    ploting the figers
    :param title: title for the figure
    :param data_set: data
    :param labels: labels for the noise vectors to map by,, if NONE just plot a regular GAN
    """
    n = 15

    noise = tf.random.normal([n, noise_dim], seed=9)
    if labels is not None:
        data_labels=labels[:n]
        fake_vecs = genrator.predict([noise,data_labels])
        fake_images = decoder.predict(fake_vecs)
    else:
        fake_vecs = genrator.predict(noise)
        fake_images = decoder.predict(fake_vecs)
    iter=data_set.__iter__()
    latent_codes = iter.__next__()[:n,:]
    decoded_imgs = decoder.predict(latent_codes)

    plt.figure(figsize=(30, 8))
    plt.suptitle(title,fontsize=30)
    for i in range(1, n):
        ax = plt.subplot(2, n, i)
        plt.imshow(decoded_imgs[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(fake_images[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def get_gan_data():
    """
    get data
    :return: data set of the vectors and labels
    """
    x_train, y_train, x_test, y_test=load_dataset()
    latent_codec = encoder(x_train)
    data_set = tf.data.Dataset.from_tensor_slices(latent_codec).batch(BATCH_SIZE)
    return data_set,y_train


###set up genral setup
gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')
disc_accuracy=tf.keras.metrics.BinaryAccuracy(name="disc acc")
gen_accuracy=tf.keras.metrics.BinaryAccuracy(name="genrator_acc")
cross_entropy = BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0006)

###set up autoencoder
encoder = build_encoder()
decoder = build_decoder()
autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.load_weights(AE_path)

###build GAN arch



data_set, train_y = get_gan_data()

##build data set
if is_CGAN:
    genrator = build_Cgenrator()
    disc = build_Cdiscriminator()
    train_gan(data_set, train_y)
    plot_figs("finale Gan",data_set,labels=train_y)

else:
    genrator = build_genrator()
    disc = build_discriminator()
    if load_w:
        genrator.load_weights("genrator_save/genrator_weights")
    else:
        train_gan(data_set)
    plot_figs("finale Gan",data_set)


if save_w:
    genrator.save_weights("genrator_save/genrator_weights")
