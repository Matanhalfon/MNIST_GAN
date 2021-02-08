from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from util import build_genrator,build_encoder,build_decoder,load_dataset,AE_path,noise_dim
import numpy as np
import os

latent_dim = 64
noise_sigma = 0.35
train_AE = False
plot_images=True
interpolate=True
sml_train_size = 50


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
	autoencoder.load_weights(AE_path)
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
		checkpoint_dir = os.path.dirname(AE_path)
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=AE_path,
														 save_weights_only=True)
		autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
						epochs=15,
						batch_size=128,
						shuffle=True,
						validation_data=(x_test, x_test),
						callbacks=[cp_callback])
	else:
		autoencoder.load_weights(AE_path)

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
	autoencoder.load_weights(AE_path)
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

