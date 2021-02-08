from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from util import build_genrator,build_encoder,build_decoder,load_dataset,checkpoint_path,noise_dim
import numpy as np
import os

latent_dim = 64
noise_sigma = 0.35
train_AE = False
plot_images=False
interpolate=True
sml_train_size = 50
#
#
# def load_dataset():
# 	# load train and test images, and pad & reshape them to (-1,32,32,1)
# 	(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
# 	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
# 	x_train = np.pad(x_train, ((0,0),(2, 2), (2, 2),(0,0)),mode='constant')
# 	x_test = np.pad(x_test, ((0,0),(2, 2), (2, 2),(0,0)),mode='constant')
# 	print(x_train.shape)
# 	print(x_test.shape)
# 	y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
# 	y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
# 	return x_train,y_train,x_test,y_test
#
#
# def build_encoder():
# 	encoder = Sequential()
# 	encoder.add(layers.Conv2D(16, (4, 4), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,1)))
# 	encoder.add(layers.Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
# 	encoder.add(layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
# 	encoder.add(layers.Conv2D(96, (3, 3), strides=(2,2), activation='relu', padding='same'))
# 	encoder.add(layers.Reshape((2*2*96,)))
# 	encoder.add(layers.Dense(latent_dim))
# 	return encoder
#
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
# def build_decoder():
# 	decoder = Sequential()
# 	decoder.add(layers.Dense(2*2*96,activation='relu', input_shape=(latent_dim,)))
# 	decoder.add(layers.Reshape((2,2,96)))
# 	decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
# 	decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
# 	decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2,2), activation='relu', padding='same'))
# 	decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2,2), activation='sigmoid', padding='same'))
# 	return decoder
#

# Your code starts here:

# Classifer Network - currently minimal
def build_classifiar():
	"""
	build an MLP classifiar over the the latent vector
	:return:
	"""
	classifier = Sequential()
	classifier.add(layers.Dense(256, activation='tanh'))
	classifier.add(layers.Dropout(rate=0.3))
	classifier.add(layers.Dense(128, activation='tanh'))
	classifier.add(layers.Dense(10,activation='softmax', input_shape=(latent_dim,)))
	return classifier


def sec2():
	"""
	run section 2, with trained and untrained encoder
	"""
	x_train, y_train, x_test, y_test =load_dataset()
	classifier=build_classifiar()
	encoder = build_encoder()
	decoder = build_decoder()
	autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
	autoencoder.load_weights(checkpoint_path)
	train_codes = encoder.predict(x_train[:sml_train_size])
	test_codes = encoder.predict(x_test)
	classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	classifier.fit(train_codes, y_train[:sml_train_size],
						epochs=200,
						batch_size=16,
						shuffle=True,
						validation_data=(test_codes, y_test))

	print("##### full classifier")

	full_cls_enc = keras.models.clone_model(encoder)
	full_cls_cls = keras.models.clone_model(classifier)
	full_cls = keras.Model(full_cls_enc.inputs,full_cls_cls(full_cls_enc.outputs))

	full_cls.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	full_cls.fit(x_train[:sml_train_size], y_train[:sml_train_size],
						epochs=100,
						batch_size=16,
						shuffle=True,
						validation_data=(x_test, y_test))


def preform_AE ():
	"""train a AE"""
	x_train,y_train,x_test,y_test=load_dataset()
	encoder=build_encoder()
	decoder=build_decoder()
	autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


	if train_AE:
		checkpoint_dir = os.path.dirname(checkpoint_path)
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
														 save_weights_only=True)
		autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
						epochs=15,
						batch_size=128,
						shuffle=True,
						validation_data=(x_test, x_test),
						callbacks=[cp_callback])
	else:
		autoencoder.load_weights(checkpoint_path)

	if plot_images:
		decoded_imgs = autoencoder.predict(x_test)
		latent_codes = encoder.predict(x_test)
		decoded_imgs = decoder.predict(latent_codes)

		n = 12
		plt.figure(figsize=(20, 4))
		for i in range(1, n + 1):
			# Display original
			ax = plt.subplot(2, n, i)
			plt.imshow(x_test[i].reshape(32, 32))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# Display reconstruction
			ax = plt.subplot(2, n, i + n)
			plt.imshow(decoded_imgs[i].reshape(32, 32))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.show()


def interpolate_vecs(v1,v2,alpha):
	for step in np.arange(0,1,alpha):
		yield v1*step+v2*(1-step)



def interpolate(is_gan=False):
	"""
	interpolate  two vectors of the latent space
	:param is_gan:  use the Gan latent space
	"""
	encoder = build_encoder()
	decoder=build_decoder()
	autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
	autoencoder.load_weights(checkpoint_path)
	if is_gan:
		genrator=build_genrator()
		genrator.load_weights("genrator_save/genrator_weights")
		noise = tf.random.normal([2, noise_dim])
		v1=tf.reshape(noise[0],(1, -1))
		v2=tf.reshape(noise[1],(1, -1))
	else:
		x_train, y_train, x_test, y_test = load_dataset()
		latent_codes = encoder.predict(x_test[:12])
		v1 = latent_codes[1].reshape((1, -1))
		v2 = latent_codes[7].reshape((1, -1))

	if is_gan:
		v1_image = decoder.predict(genrator.predict(v1))
		v2_image = decoder.predict(genrator.predict(v2))
	else:
		v1_image = decoder.predict(v1)
		v2_image = decoder.predict(v2)
	plt.figure(figsize=(20,4 ))
	plt.subplot(1, 12, 1)
	plt.imshow(v1_image.reshape((32, 32)))
	plt.title("here v1")
	plt.gray()
	for i,vec in enumerate(interpolate_vecs(v2,v1,0.1)):
		if is_gan:
			interpolated = decoder.predict(genrator.predict(vec))
		else:
			interpolated = decoder.predict(vec)
		plt.subplot(1,12,2+i)
		plt.imshow(interpolated.reshape((32,32)))
		plt.gray()

	plt.subplot(1, 12, 12)
	plt.imshow(v2_image.reshape((32,32)))
	plt.title("here v2")
	plt.gray()
	plt.show()

