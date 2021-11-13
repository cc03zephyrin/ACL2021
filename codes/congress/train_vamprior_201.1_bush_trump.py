import itertools
import matplotlib as mpl
import numpy as np
import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.layers import dense
from tensorlayer.activation import hard_tanh

tf.disable_v2_behavior() 
# import tensorflow.contrib.slim as slim
import tf_slim as slim
import time
# import seaborn as sns
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
import scipy

# from tensorflow.contrib.layers import fully_connected
from matplotlib import pyplot as plt
import tensorflow_probability as tfp
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score


# import warnings

# def fxn():
# 	warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
# 	warnings.simplefilter("ignore")
# 	fxn()

# from imageio import imwrite
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# sns.set_style('whitegrid')

# distributions = tf.distributions
vs = 201.11 # vamprior, adding masked labels, randomly masked
penalty = 1


flags =tf.compat.v1.flags
flags.DEFINE_bool('biased', True, 'enabled biased sampling')
flags.DEFINE_bool('remove_history', True, 'remove training history logs and models')
flags.DEFINE_bool('iter', True, 'do iterative training')

flags.DEFINE_bool('tuning', True, "tuning parameters")
flags.DEFINE_float('notmask', 0.05, 'non_masked label')

# flags.DEFINE_string('data_dir', r'C:\code\chen_proj_codes\news_bias\congressionalreport\processed\congress_report_topics\data_for_train', 'Directory for data')
# flags.DEFINE_float('beta_the', 1, 'coef of theta ')
flags.DEFINE_float('beta_kl', 1, 'coef for the kl diveregence of the priors')
flags.DEFINE_float('beta_th', 5.0, 'coef of theta constraints to original theta0 set to 1,') #  this is actual number / batchsize *2
flags.DEFINE_float('beta_t', 5.0, 'coef of constraints of T close to T0 or T00')
flags.DEFINE_float('beta_z', 15.0, 'z_mu stability, close to zbar')
flags.DEFINE_float('beta_p', 2, 'topic vector orthogonal to polarization')
flags.DEFINE_float('beta_s', 10.0, 'prediction of slants')
flags.DEFINE_float('beta_mse',250.0, 'coeff of mse, if standard normal set to 250')
flags.DEFINE_integer('alter_rounds', 5, 'alternate rounds') #  default 10
flags.DEFINE_integer('n_epochs', 10, 'number of epochs') # default 30
flags.DEFINE_float("pseudu_mean", 0.02, 'expectation of vamprior')# set to a value close to 0
flags.DEFINE_float('pseudu_std', 0.3, 'std of vamprior')
flags.DEFINE_integer('num_comp', 2, 'number of Gaussian mixtures, default 2')
flags.DEFINE_float('anneal_kl', 20, 'turn off regulization until KL reaches high value') #mean 6 large 13
flags.DEFINE_bool('Anneal', True, 'if true, turn off regulization until KL reaches high value')



# For bigger model:
flags.DEFINE_string('data_dir', r'/home/cczephyrin/projects/political embedding/training_vec/', 'Directory for data')
flags.DEFINE_string('logdir', r'/home/cczephyrin/projects/political embedding/logs/vae_semi_v{}/'.format(vs), 'Directory for logs')
flags.DEFINE_string('resultdir', r'/home/cczephyrin/projects/political embedding/results/frames/framing_train_v{}/'.format(vs), 'directory for results')
flags.DEFINE_string('modelpath', r'/home/cczephyrin/projects/political embedding/models/vae_semi_v{}/'.format(vs), 'Directory for models')
flags.DEFINE_integer('latent_dim', 50, 'Latent dimensionality and output dim of model') #modified by XR, change dimensionality of z to 2.
flags.DEFINE_integer('latent_output_dim', 300, 'output dim and input dim of model')  ### Chen: added negative output dimension
flags.DEFINE_integer('batch_size', 500, 'Minibatch size')
# flags.DEFINE_integer('n_samples', 1, 'Number of samples to save')
flags.DEFINE_integer('print_every', 50, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 800, 'Hidden size for neural networks')

flags.DEFINE_integer('n_topics', 68, 'number of topics')

flags.DEFINE_integer('n_authors', 1674, 'number of authors')
flags.DEFINE_integer('bottom', 0, 'start from a fraction')
FLAGS = flags.FLAGS



def PCA_mutual_info(Zbar, party_label, As):

	scaler = StandardScaler()


	# select non-zero rows
	nonzero_inx = (np.sum(As.T, axis = -1) !=0)
	Zbar_nonzero = Zbar[nonzero_inx]
	party_true = party_label[nonzero_inx]
	Zbar_nonzero = scaler.fit_transform(Zbar_nonzero)

	# test top 1 component
	pca = PCA(n_components=1)
	Zbar_pca = pca.fit_transform(Zbar_nonzero)
	# pty_pred = KMeans(n_clusters=2).fit_predict(Zbar_pca)
	gm = GMM(n_components=2).fit(Zbar_pca)
	pty_pred = gm.predict(Zbar_pca)
	print('##########*********calculate mutual info for top 1 components****************')
	print(np.mean(pty_pred==party_true))
	print(np.mean(pty_pred!=party_true))

	# print(" clustering attributes mean variance and weights for one component")
	# print(gm.means_)
	# print(gm.covariances_)
	# print(gm.weights_)
	mutualinfo1 = mutual_info_score(pty_pred, party_true)
	print("PCA1 mi", mutualinfo1)

	# test top 2 componnet

	pca = PCA(n_components=2)
	Zbar_pca = pca.fit_transform(Zbar_nonzero)
	# pty_pred = KMeans(n_clusters=2).fit_predict(Zbar_pca)
	gm = GMM(n_components=2).fit(Zbar_pca)
	pty_pred = gm.predict(Zbar_pca)
	print('##########*********calculate mutual info for top 2 components****************')
	print(np.mean(pty_pred==party_true))
	print(np.mean(pty_pred!=party_true))

	# print(" clustering attributes mean variance and weights for two components")
	# print(gm.means_)
	# print(gm.covariances_)
	# print(gm.weights_)
	mutualinfo2 = mutual_info_score(pty_pred, party_true)
	print("PCA2 mi", mutualinfo2)


	# PCA and Clustering 
	pca = PCA(n_components=5)
	Zbar_pca = pca.fit_transform(Zbar_nonzero)
	# pty_pred = KMeans(n_clusters=2).fit_predict(Zbar_pca)
	gm = GMM(n_components=2).fit(Zbar_pca)
	pty_pred = gm.predict(Zbar_pca)
	print('##########*********calculate mutual info for top 5 components****************')
	print(np.mean(pty_pred==party_true))
	print(np.mean(pty_pred!=party_true))

	# print(" clustering attributes mean variance and weights")
	# print(gm.means_)
	# print(gm.covariances_)
	# print(gm.weights_)
	mutualinfo3 = mutual_info_score(pty_pred, party_true)
	print("PCA5 mi", mutualinfo3)
	
	return np.amax([mutualinfo1, mutualinfo2,mutualinfo3])

def predict_party_membership(s_pred, y, As):
	nonzero_inx = (np.sum(As.T, axis = -1) !=0)
	ave_slants = (np.matmul(As.T, s_pred)/(np.sum(As.T, axis = -1) + 0.000001))[nonzero_inx]
	y_true = y[nonzero_inx]
	y_pred = ave_slants>0
	return np.mean(y_true == y_pred)


def custom_hardtanh(x, cmin= -6.0, cmax = 3.0, name = 'htanh'):
	return tf.clip_by_value(x, cmin, cmax, name = name)


def custom_crossentrophy(x, z):
	return tf.math.maximum(x, 0) - x * z + tf.math.log(1 + tf.math.exp(-tf.math.abs(x)))



def framing_inference_network(author, topic,d, latent_dim, hidden_size,trainable = True):
	"""Construct an inference network parametrizing a Gaussian.
	Args:
	x: A batch of MNIST digits.
	latent_dim: The latent dimensionality.
	hidden_size: The size of the neural net hidden layers.
	Returns:
	mu: Mean parameters for the variational family Normal
	sigma: Standard deviation parameters for the variational family Normal

	"""
	# nonlinear = custom_hardtanh(min= -6.0, max = 3.0)
	with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, trainable = trainable):
		# author = tf.cast(slim.flatten(author), tf.float32)
		# topic = tf.cast(slim.flatten(topic), tf.float32)
		# d = slim.flatten(d)
		# author = tf.to_float(author)
		# topic = tf.to_float(topic)
		# at = tf.concat([author, topic], -1)

		# net1 = slim.fully_connected(at, 2 * hidden_size)
		# net1 = slim.fully_connected(net1, hidden_size)
		net0 = slim.fully_connected(d, hidden_size)
		net0 = slim.fully_connected(net0, hidden_size)

		net_mu = slim.fully_connected(net0, hidden_size)
		net_mu= slim.fully_connected(net_mu, int(hidden_size/2))
		mu = slim.fully_connected(net_mu, latent_dim, activation_fn = None)  ### here mean should be able to take negative value

		net_sigma = slim.fully_connected(net0, hidden_size)
		net_sigma= slim.fully_connected(net_sigma, int(hidden_size/2))
		# The standard deviation must be positive,.Parameterize with a softplus
		# sigma = tf.nn.softplus(slim.fully_connected(net_sigma, latent_dim, activation_fn=None))
		logsigma2 = slim.fully_connected(net_sigma, latent_dim, activation_fn= lambda x: custom_hardtanh(x, cmin= -6.0, cmax = 3.0))
	return mu, logsigma2




def gen_vampriors(num_comp, input_dim, trainable = True):
	# creat idle input

	idle = tf.cast(tf.constant(np.eye(num_comp)), dtype= tf.float32) 
	# nonlinear = custom_hardtanh(min = -0.4, max = 0.6)

	init = tf.random_normal_initializer(mean=FLAGS.pseudu_mean, stddev=FLAGS.pseudu_std)
	# idle = tf.get_variable("psuedo_input_gen",dtype=tf.float32, initializer=tf.constant(np.eye(num_comp), dtype = tf.float32), trainable = False)
	# pseudip_w= tf.get_variable("pseudoip_weight",shape = [5,300], dtype=tf.float32, trainable = True)
	# create initiator
	

	# create denselayer, output vampriors, of size C * input_dim 300
	vps = dense(idle, units = input_dim, kernel_initializer = init, use_bias = False, activation = lambda x: custom_hardtanh(x, cmin= -0.4, cmax = 0.6))
	# vps = hard_tanh(tf.matmul(idle, pseudip_w))

	return vps
	

def log_Normal_diag(x, mean, logvar, dim=None):

	log_normal = -0.5 * (logvar + tf.pow( x - mean, 2 ) / tf.math.exp( logvar ) )

	return tf.reduce_sum( log_normal, dim )


def log_prior_z(z_sample, z_p_mu, z_p_logvar):
	# expand z
	z_expand = tf.expand_dims(z_sample, 1)
	means = tf.expand_dims(z_p_mu, 0)
	logvars = tf.expand_dims(z_p_logvar, 0)
	# means = z_p_mu
	# logvars= z_p_logvar
	C = FLAGS.num_comp
	a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)
	  # batch x C
	a_max= tf.math.reduce_max(a, 1)  # batch size, preventing overflow
	# calculte log-sum-exp
	log_prior = (a_max + tf.math.log(tf.reduce_sum(tf.math.exp(a - tf.expand_dims(a_max, 1)), 1)))  # batch size
	return log_prior
# edited by XR

def framing_inference_network_decoder(z, hidden_size, latent_output_dim, trainable=True):
	"""Decoder of the framing inference network
	Args:
		z: the latent variable (input)
		hidden_size: The size of the neural net hidden layers
		latent_output_dim: dim of the decoder's output
	Returns:
		A latent variable decoded
	"""

	with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, trainable=trainable):
		net_decoder = slim.fully_connected(z, hidden_size)
		net_decoder = slim.fully_connected(net_decoder, hidden_size) ### Chen:add a new layer to increase the decoding power
		net_decoder = slim.fully_connected(net_decoder, latent_output_dim, activation_fn = None) ### Chen:again output should take negative values
	return net_decoder


def topic_inference_network(d,topic_dim, hidden_size):
	"""Construct an inference network parametrizing a Gaussian.
	Args:
	x: A batch of document vectors.
	  topic_dim: The latent dimensionality.
	  hidden_size: The size of the neural net hidden layers.
	Returns:
	  mu: Mean parameters for the variational family Normal
	  sigma: Standard deviation parameters for the variational family Normal
	"""
	with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		# net = slim.flatten(x)
		d = tf.to_float(d)
		net = slim.fully_connected(d, hidden_size)
		net = slim.fully_connected(net, hidden_size*2)
		net = slim.fully_connected(net, int(hidden_size/2))
		gaussian_params = slim.fully_connected(
			net, topic_dim, activation_fn=None)
		# The mean parameter is unconstrained
		# mu = gaussian_params[:, :topic_dim]
		# # The standard deviation must be positive. Parametrize with a softplus
		# sigma = tf.nn.softplus(gaussian_params[:, topic_dim:])
	return gaussian_params

def slant_network(z, hidden_size, output_dim = 1):
	"""Build a generative network parametrizing the likelihood of the data
	Args:
	z: Samples of latent variables
	hidden_size: Size of the hidden state of the neural net
	Returns:
	bernoulli_logits: logits for the Bernoulli likelihood of the data
	"""
	with slim.arg_scope([slim.fully_connected], activation_fn= tf.nn.relu):
		z = tf.to_float(z)
		net = slim.fully_connected(z, hidden_size)
		net = slim.fully_connected(net, hidden_size)
		output = slim.fully_connected(net, output_dim, activation_fn = None)
	# bernoulli_logits = tf.reshape(bernoulli_logits, [-1, 28, 28, 1])
	return output


def model(TrainD, TrainA, TrainThe, TrainID, TrainParty, TrainMask, trained = True, reuse = False, Zbar = None):
	trainf = True
	traint = True 


	with tf.name_scope('data'):

		batchno = tf.Variable(0, dtype=tf.int64)
		TrainParty = tf.cast(TrainParty, tf.float32)
		if Zbar is not None:
			Zbar = tf.cast(tf.constant(Zbar), tf.float32) 
		else:
			Zbar = tf.constant(np.random.normal(size = (FLAGS.n_authors, FLAGS.latent_dim))).astype(np.float32)

		zbar = tf.gather(Zbar, tf.argmax(TrainA, -1))
		Mask = tf.cast(TrainMask, tf.float32)
		# print('zbar shape!!!!',tf.shape(zbar))




	with tf.variable_scope('variational_f', reuse = reuse):
		# if test:
		# 	d= TrainDr
		# else:
		# 	d = TrainD
		z_mu, z_logsig2 = framing_inference_network(TrainA, TrainThe, TrainD, latent_dim=FLAGS.latent_dim,
												  hidden_size=FLAGS.hidden_size)
		# z_var = tf.pow(z_sigma, 2)
		# log_z_var = tf.math.log(z_var)
		# The variational distribution is a Normal with mean and standard
		# deviation given by the inference network
		# f_z = distributions.Normal(loc=f_mu, scale=f_sigma) # draw the consistent framing

		z_sigma = tf.exp(z_logsig2/2)

		# The variational distribution is a Normal with mean and standard
		# deviation given by the inference network
		z_dist = tfp.distributions.Normal(loc=z_mu, scale=z_sigma)


		# assert ft_z.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED
		# get framing vector
		
		# f = ft_mu	
		z = z_mu #edited by XR
		z_sample = z_dist.sample()
		# z_theta = tf.concat((z_sample, theta_n), axis=-1)
		# print('z_mu shape!!!', tf.shape(z_mu))
		
		


	with tf.variable_scope('slant_pred', reuse = reuse):
		
		s_logit = slant_network(z_mu, FLAGS.hidden_size, output_dim = 1)
		s_logit = tf.squeeze(s_logit, axis = 1)
		s_pred = tf.cast(s_logit > 0, tf.int64)
	 # latent dim 300
	# get polarization vect
	# p_t = tf.matmul(theta_n, P)  # 67 and 67* 300
	with tf.variable_scope('model', reuse = reuse):
		# The likelihood is Bernoulli-distributed with logits given by the
		# generative network
		
		# d_recon= ft #edited by XR: change ft to f
		f = framing_inference_network_decoder(z_sample, hidden_size=FLAGS.hidden_size, latent_output_dim= FLAGS.latent_output_dim) ### Chen: use flag for consistency

		p_x_given_z = f

	# Generate vampriors, of size C * inputdim
	with tf.variable_scope('vamprior', reuse = reuse):
		vamps = gen_vampriors(FLAGS.num_comp, FLAGS.latent_output_dim) #C * 300
		vamps = tf.cast(vamps, tf.float32)
		# print('vamps  shape ', tf.shape(vamps))
		# vamp = tf.reduce_mean(vamps)

	with tf.variable_scope('variational_f', reuse = True):

		z_p_mu, z_p_logvar = framing_inference_network(TrainA, TrainThe, vamps, latent_dim=FLAGS.latent_dim,
												  hidden_size=FLAGS.hidden_size)
		# z_p_var = tf.pow(z_p_sigma, 2)
		# z_p_logvar = tf.math.log(z_p_var)
		

	#
	# Build the evidence lower bound (ELBO) or the negative loss
	if trained:
		p_z_given_x = log_Normal_diag(z_sample, z_mu, z_logsig2, dim =1)
		p_z_given_v = log_prior_z(z_sample, z_p_mu, z_p_logvar)
		kl_mu_pre  = (- p_z_given_v+ p_z_given_x)# kl divergence for gmm vampriors
		# if anneal is true
		if FLAGS.Anneal:
			# maxi = tf.cast(FLAGS.anneal_kl, tf.int64)
			kl_mu = tf.math.maximum(FLAGS.anneal_kl, kl_mu_pre)
		else:
			kl_mu = kl_mu_pre

		mse = FLAGS.beta_mse* tf.losses.mean_squared_error(TrainD, p_x_given_z)/2  # assuming the variance is 1
		zmuLoss = FLAGS.beta_z * tf.losses.mean_squared_error(z_mu, zbar)
		SlantLoss = FLAGS.beta_s * Mask * custom_crossentrophy(s_logit, TrainParty)


		elbo = tf.reduce_mean(- mse -kl_mu   - zmuLoss -SlantLoss, 0)


		optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

		train_op = optimizer.minimize(-elbo)
		# if writesummary:
		elbo_loss = tf.summary.scalar('elbo_m', tf.reduce_mean(elbo))
		mse_loss = tf.summary.scalar('mse_m', tf.reduce_mean(mse))
		kl_mu_loss = tf.summary.scalar('kl_mu_gmmvampriors', tf.reduce_mean(kl_mu))

		zmu_loss = tf.summary.scalar('zmu oppose to zbar loss', tf.reduce_mean(zmuLoss))
		z_v_prior = tf.summary.scalar('z prob given pseudo input', tf.reduce_mean(p_z_given_v))
		z_x_prior = tf.summary.scalar('z prob given actual input', tf.reduce_mean(p_z_given_x))
		# vamp_check = tf.summary.scalar('check vampriors', tf.reduce_mean(vamps))

        #below edited by XR
		S_loss = tf.summary.scalar('Predicted slant oppose to party slant loss', tf.reduce_mean(SlantLoss))
		for v in tf.trainable_variables():
			tf.summary.scalar(v.name, tf.reduce_mean(v))

		summary_op = tf.summary.merge_all()
		# summary_ops = [summary_op, elbo, mse, kl_ft, train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,
		#			   batchno.assign_add(1)] #edited by XR: change f to z, ft to f
		summary_ops = [summary_op, elbo, mse, kl_mu, train_op, TrainID, TrainParty,  z, f,vamps,  batchno.assign_add(1)]
			# print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)

		# Merge all the summaries

		# train_ops = [train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,batchno.assign_add(1)]
		train_ops = [train_op, TrainID, TrainParty, z, f, vamps,  kl_mu, batchno.assign_add(1)] # edited by XR: delete f, change ft to f


		return train_ops, summary_ops, 

	else:
		mse = FLAGS.beta_mse* tf.losses.mean_squared_error(TrainD, p_x_given_z)/2.0
		zmuLoss = FLAGS.beta_z * tf.losses.mean_squared_error(z_mu, zbar)

		zmu_loss_val = tf.summary.scalar('zmu oppose to zbar loss in val, some zbar will be 0', tf.reduce_mean(zmuLoss))
		mse_loss_val = tf.summary.scalar('mse_val', tf.reduce_mean(mse))
		# summary_op = tf.summary.merge([mse_loss_val, F_loss_val, Theta_loss_val, T_loss_val]) #edited by XR
		summary_op = tf.summary.merge([mse_loss_val, zmu_loss_val])
		# val_ops = [summary_op, TrainID, TrainParty, theta_n, TrainThe, ft, TrainA, mse, s_pred, batchno.assign_add(1)] 
		val_ops = [summary_op, TrainID, TrainParty,  z,f, vamps, z_p_mu, z_p_logvar, TrainA, mse, s_logit, batchno.assign_add(1)] # edited by XR: change ft to f

	
		return val_ops



def train(trains, vals,  tests, Zbar = None,  updateT = None, round = 0, do_val = False, writesummary = False):
	# Train a Variational Autoencoder				batchcount = 0

	# for each round, update thetas, and P0, T0

	# Input placeholders


	Ds, As, Thetas, Parties, Ids, Masks = trains

	Ds_val, As_val, Thetas_val, Parties_val, Ids_val, Masks_val= vals
	Ds_test, As_test, Thetas_test, Parties_test, Ids_test, Masks_test= tests
	print(Ds.shape)
	print(As.shape)

	print(Thetas.shape)
	Ds = np.float32(Ds)
	Ds_val = np.float32(Ds_val)
	Ds_test = np.float32(Ds_test)

	C = FLAGS.num_comp

	with tf.Graph().as_default():

		Dt = tf.placeholder(tf.float32, shape=[None,300])
		At = tf.placeholder(tf.int64, shape = [None, As.shape[-1]])
		Thet = tf.placeholder(tf.float32, shape = [None, Thetas.shape[-1]])
		Ptr = tf.placeholder(tf.int64, shape = [None,])
		Idt = tf.placeholder(tf.int64, shape= [None, ])
		Maskt = tf.placeholder(tf.int64, shape = [None,])

		train_dataset = tf.data.Dataset.from_tensor_slices((Dt, At, Thet,  Idt, Ptr, Maskt))
		# train_dataset = train_dataset.shuffle(buffer_size = 100000)
		train_dataset = train_dataset.batch(FLAGS.batch_size)

		train_iterator = train_dataset.make_initializable_iterator()
		TrainD, TrainA, TrainThe, TrainID, TrainParty, TrainMask= train_iterator.get_next()

		tf.cast(TrainD, tf.float32)
		tf.cast(TrainA, tf.float32)

		val_dataset = tf.data.Dataset.from_tensor_slices((Ds_val, As_val, Thetas_val,  Ids_val, Parties_val, Masks_val))
		val_dataset = val_dataset.batch(FLAGS.batch_size)
		val_iterator = val_dataset.make_initializable_iterator()
		ValD, ValA, ValThe, ValID, ValParty, ValMask= val_iterator.get_next()

		test_dataset = tf.data.Dataset.from_tensor_slices((Ds_test, As_test, Thetas_test,  Ids_test, Parties_test, Masks_test))
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		TstD, TstA, TstThe, TstID, TstParty, TstMask= test_iterator.get_next()
		# tf.cast(TrainD, tf.float32)


		train_ops, summary_ops = model(TrainD, TrainA, TrainThe, TrainID, TrainParty, TrainMask, Zbar = Zbar)
		saver = tf.train.Saver()
		val_ops= model(ValD, ValA, ValThe, ValID, ValParty, ValMask, trained = False, reuse = True,  Zbar = Zbar)
		test_ops = model(TstD, TstA, TstThe, TstID, TstParty, TstMask, trained = False, reuse = True,  Zbar = Zbar)

		init_op = tf.global_variables_initializer()
		# with tf.InteractiveSession() as sess:
		with tf.Session() as sess:
			sess.run(init_op)
			# if writesummary:
			train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train/{}/'.format(round), sess.graph)
			print('start training')
			for epoch in range(FLAGS.n_epochs):
				print("start epoch {}".format(epoch))

				sess.run(train_iterator.initializer, feed_dict = {Dt:Ds, At:As, Thet:Thetas, Ptr:Parties, Idt:Ids, Maskt: Masks})
				# print(sess.run([TrainD, TrainParty, TrainThe]))


				speechids = []
				# topics_mix = []
				Zmus = []
				Frames = []
				# elbos = []
				# mses = []
				# kl_fs= []
				# kl_ts = []
				parties = []
				# topics0_mix = []
				kl_mus = []

				batchcount = 0
				# vamps_= []
				while True:
					try:
						if not writesummary: # only write summary for some rounds
							# _, TrainID_, TrainParty_,T_, theta_n_, theta0_, f_, ft_, batchcount= sess.run(train_ops) # Ds, As, Thetas, Parties,Ids
							# edited by XR, remove f_ and change ft_ to f_
							_, TrainID_, TrainParty_, z_, f_, vamps, _,batchcount= sess.run(train_ops) # Ds, As, Thetas, Parties,Ids
							speechids.extend(TrainID_)
							
							# topics_mix.extend(theta_n_)
							Zmus.extend(z_) #edited by XR: change f_ to z_
							Frames.extend(f_)
							parties.extend(TrainParty_)
							# topics0_mix.extend(theta0_)
							# print('vamps', vamps)
							# vamps_.extend(vamps)
							# print('Trains', z_, theta_n_, TrainParty_)
							# break

						else:
							if (batchcount+1)%FLAGS.print_every ==0:
								# print(batchcount)
								# print('write summpary')

								# summary_str, elbo_, mse_, kl_f_,  _, TrainID_, TrainParty_,T_, theta_n_, theta0_, f_, ft_, batchcount = sess.run(summary_ops)
								summary_str, elbo_, mse_, kl_mu_,  _, TrainID_, TrainParty_, z_, f_, vamps, batchcount = sess.run(summary_ops) # edited by XR: change f_ to z_ and ft_ to f_
								# summary_str, _, _, _, _ = sess.run(summary_ops)
								# elbos.extend(elbo_)
								# mses.append(mse_)
								# kl_fs.extend(kl_f_)
								# kl_ts.extend(kl_t_)
								kl_mus.extend(kl_mu_)

								### test vamps######
								# print('vamps', vamps, vamps.shape,zmu)
								# vamps_.extend(vamps)
								# print('Trains', z_, theta_n_, TrainParty_)
								# break

								######################
								train_writer.add_summary(summary_str, batchcount)
								speechids.extend(TrainID_)
								
								Zmus.extend(z_)
								Frames.extend(f_)

								
								parties.extend(TrainParty_)
							else:
								# _, TrainID_, TrainParty_, T_, theta_n_, theta0_, f_, ft_, batchcount = sess.run(
								# 	train_ops)  # Ds, As, Thetas, Parties,Ids
								# edited by XR: delete ft_
								_, TrainID_, TrainParty_,  z_, f_, vamps, _,batchcount = sess.run(
									train_ops)  # Ds, As, Thetas, Parties,Ids
								speechids.extend(TrainID_)
							
								Zmus.extend(z_)
								Frames.extend(f_)
								parties.extend(TrainParty_)

								### #######################test vamps#############################
								# print('vamps and sigma2', vamps, vamps.shape, zmu)
								# vamps_.extend(vamps)
								# print('Trains', z_, theta_n_, TrainParty_, Zmus[0])
								# break
								#########################################################################

					except tf.errors.OutOfRangeError:
						kl_mus = np.array(kl_mus)
						print("average, max kl_mu at epoch {}".format(epoch), np.mean(kl_mus), np.amax(kl_mus))
						print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
						break
						# start test
			print("begin test")
			tst_writer = tf.summary.FileWriter(FLAGS.logdir + '/test/{}/'.format(round), sess.graph)
			sess.run(test_iterator.initializer)
			tst_ids =[]
			tst_slants = []
			tst_Zmus = []
			tst_thetas = []
			tst_parties = []
			while True:
				try:
					summary_tst_str, TstID_, TstParty_, z_, _, _, _, _, _, _, S_pred_,batchcount = sess.run(test_ops) 
					tst_ids.extend(TstID_)
					tst_slants.extend(S_pred_)
					tst_Zmus.extend(z_)
					
					tst_parties.extend(TstParty_)
				except tf.errors.OutOfRangeError:
					break
			tst_ids = np.array(tst_ids)
			tst_slants = np.array(tst_slants)
			tst_Zmus = np.array(tst_Zmus)
			tst_parties = np.array(tst_parties)
			assert (tst_ids==Ids_test).all(), 'test ids no match, check id order to make sure order is not changed'
			assert (tst_parties== Parties_test).all(), 'test party no match, check party order to make sure order is not changed'



			if do_val:
				print('beging evaluation')
				val_writer = tf.summary.FileWriter(FLAGS.logdir + '/val/{}/'.format(round), sess.graph)
				sess.run(val_iterator.initializer)
				# topics_val = []
				val_ids = []
				val_Zmus = []
				val_Frames = []
				val_parties = []
				val_mse = []
				val_authors = []
				val_slants = []
				# topics0_val = []
				# val_Z_pmus = []
				while True:
					try:
						summary_val_str, ValID_, ValParty_,  z_, f_, Vampriors, Zp_mus, Zp_sigmas, ValA_,  mse_, S_pred_,batchcount = sess.run(val_ops) # edited by XR: change ft_ to f_, remove S_pred_
						val_writer.add_summary(summary_val_str, batchcount)
						val_Zmus.extend(z_)
						val_Frames.extend(f_)
						val_ids.extend(ValID_)
						val_parties.extend(ValParty_)
						# topics_val.extend(theta_n_)
						val_mse.append(mse_)
						val_authors.extend(ValA_)
						# val_slants.extend(S_pred_)
						# topics0_val.extend(theta0_)
						val_slants.extend(S_pred_)
						# val_Z_pmus.extend(Zp_mus)


					except tf.errors.OutOfRangeError:
						break
				val_ids = np.array(val_ids)
				val_parties = np.array(val_parties)
				val_slants = np.array(val_slants)
				# print(val_ids.shape, val_ids[:10])
				# print(Ids_val.shape, Ids_val[:10])
				print('mse in val')
				print(np.mean(val_mse))
				val_authors = np.argmax(np.array(val_authors), axis = 1)
				val_Zmus = np.array(val_Zmus)


				assert (val_ids==Ids_val).all(), 'validation ids no match, check id order to make sure order is not changed'
				assert (val_parties == Parties_val).all()==True, 'validation parties no match'

				

			print('save model')
			try:
				os.mkdir(FLAGS.modelpath+'/{}/'.format(round))
			except:
				pass
			# outpus vamps, Zprior_mu and Zprior_std

			
			saver.save(sess=sess, save_path=FLAGS.modelpath+'/{}/vae_{}'.format(round, round))


	

	if do_val:
		speechids = np.array(speechids)


		# Zmus = np.array(Zmus)
		# all zp_mus in val will be the sames 
		return speechids, Zmus, Frames, parties,  val_Zmus, val_Frames, val_ids, val_parties, val_authors, Vampriors, Zp_mus, Zp_sigmas, val_slants, tst_Zmus, tst_slants #edited by XR: save Frames later
	else:
		speechids = np.array(speechids)
		# Zmus = np.array(Zmus)
		return Tmatrix, topics_mix, speechids, Zmus, Frames, parties



def iterative_training():
	# load data, initialize P0 and T0

	datapath = FLAGS.data_dir
	# crs_1991_final = pd.read_csv(r'C:\code\chen_proj_codes\\news_bias\congressionalreport\data'
	# 				   '\\all congressional speech\processed\\cong_reports_102_114_for_training.csv', sep='|',
	# 							 encoding='ISO-8859-1', index_col=0)

	# speeches =crs_1991_final['speech']

	# filter out true switchers from training sets 
	true_switchers_ = ['A000361','B000229', 'B001264', 'D000168', 'G000280', 'H000067', 'H000390', 'L000119', 'P000066', 'T000058', 'C000077','F000257', 'G000557', 
	'S000320', 'S000709','J000072']

	independents = ['S000033', 'B001237', 'K000383']

	true_switchers =independents + true_switchers_
	enddate = '2020-10-04'
	# bottom= FLAGS.bottom

	date = np.load(os.path.join(datapath, 'sorted_date_1991_{}.npy'.format(enddate)))
	
	date = np.load(os.path.join(datapath, 'sorted_date_1991_{}.npy'.format(enddate)))
	
	inx = date>20010000 # sub index data


	#########################################################################

	ds_all = np.load(os.path.join(datapath, 'cr_sentencevectors_1991_{}.npy'.format(enddate)))[inx]
	# print("mean and std of ds", np.mean(ds_all, axis = 0), np.std(ds_all, axis = 0))

	Zmus_pre = np.load(os.path.join(datapath, 'Z_mus_pre.npy'))


	# ds_all = np.load(np.load(os.path.join(datapath, 'cong_reports_102_114_for_training_sif.npz'))['arr_0'])


	# ds_all = np.load(os.path.join(datapath, 'cong_reports_102_114_for_training_sif.npz'))['arr_0'][bottom:]
	ds_all = np.float32(ds_all)
	# print(len(ds_all))

	# print(ds.shape)
	As_all = np.load(os.path.join(datapath, 'cr_author_dummy_1991_{}.npy'.format(enddate)))[inx]
	theta_dummy_all = np.load(os.path.join(datapath, 'cr_topic_dummy_1991_{}.npy'.format(enddate)))[inx]
	# is_senate = np.load(os.path.join(datapath, 'cr_chamber_is_senate_1991_{}.npy'.format(enddate)))[inx]
	# assert len(is_senate) == len(theta_dummy_all), 'is senate length error'
	theta_categ = np.argmax(theta_dummy_all, axis = -1)
	print("Author matrix shape is ", As_all.shape)
	thetas_all = np.load(os.path.join(datapath, 'cr_topic_softmax_1991_{}.npy'.format(enddate)))[inx]
	T00 = np.load(os.path.join(datapath, 'cr_Topicvectors_trump_withother_1991_{}.npy'.format(enddate)))
	# print("T0 is ",T00)
	# T00 = np.roll(T00, 1, axis = 0) # roll 1 because theta starts with -1, whle original tpv ends with -1, move the last row to first
	parties_all = np.load(os.path.join(datapath, 'cr_parties_1991_{}.npy'.format(enddate)))[inx]
	ids_all = np.load(os.path.join(datapath, 'cr_speechids_1991_{}.npy'.format(enddate)), allow_pickle= True)[inx]
	P0 = np.load(os.path.join(datapath, 'PR_axis_piror_1991_{}.npy'.format(enddate)))
	P0 = np.roll(P0, -1, axis = 0) # orignal P0 first row corresponding to the null topic, which is the last topic,
	# P0 = np.load(r'C:\code\chen_proj_codes\\news_bias\congressionalreport\processed\congress_report_topics\trial_data\PR_sample_piror.npy')
	assert P0.shape ==(68, 300), 'dr axis prior shape error'

	# thetas_all = thetas0_all.copy()
	T0 = T00.copy()
	# print('loaded data')
	# split

	# calculate Zbar0 matrix

	# Zbar0 = np.matmul(As_all.T, Zmus_pre)/(np.sum(As_all, axis = 0)[:, np.newaxis]+0.000001)
	Zbar0 = np.random.rand(As_all.T.shape[0], FLAGS.latent_dim)

	# i_all = np.array([np.eye(FLAGS.num_comp, dtype = np.float32)] * len(ds_all))
	# print('idles shape', idles_all.shape)

	As_ids = np.load(os.path.join(datapath, 'cr_author_id_1991_{}.npy'.format(enddate)),allow_pickle=True)
	all_valid = np.sum(As_all.T, axis  = 1)>0
	# print(As_all.shape)
	parties_congress = np.matmul(As_all.T, parties_all)>0
	As_can = As_ids[all_valid]
	parties_can = parties_congress[all_valid]
 	
	au_c, pt_c = zip(*[(i, j) for i, j in zip(As_can, parties_can) if i not in true_switchers])
	# print(len(au_c), len(pt_c))
	# print(au_c, pt_c)
	# exit()
	if not FLAGS.biased:
		sss = StratifiedShuffleSplit(n_splits=5, test_size= 1- FLAGS.notmask)
		r, m = next(sss.split(au_c, pt_c))
		idsrev = np.array(au_c)[r]
		idsmsk = np.array(au_c)[m]
		# print(len(idsrev), len(idsmsk))
		mskdummies = [list(As_ids).index(i) for i in idsmsk if i in As_ids]
		allvaliddummies = [list(As_ids).index(i) for i in au_c]
		train_id_wo_switchers= As_all[:, allvaliddummies].sum(axis = -1)==1
		test_id_wo_switchers = As_all[:, mskdummies].sum(axis = -1)==1 # no need ot filter swithchers

		mask_ = np.array([1]* len(parties_all))
		mask_[test_id_wo_switchers] =0
		mask_pre = mask_.copy()
	else:
		dws_all = np.float32(np.load(os.path.join(datapath, 'cr_authordw_bydoc_1991_{}.npy'.format(enddate))))[inx][:, 0] # if dw is 0 then no dw records dws by doc
		print(dws_all.shape)
		dws_id = np.matmul(As_all.T, dws_all)/(np.sum(As_all.T, axis = -1) + 0.0000001)
		orderid = np.argsort(dws_id)
		# rand ids

		As_ids_ranked = As_ids[orderid]
		As_ids_rankfilt = np.array([i for i in As_ids_ranked if i in au_c])
		tbn = int(len(As_ids_rankfilt) * FLAGS.notmask/2)

		ids_msk = As_ids_rankfilt[tbn:-tbn]
		allvaliddummies = [list(As_ids).index(i) for i in au_c]
		train_id_wo_switchers= As_all[:, allvaliddummies].sum(axis = -1)==1
		test_id_wo_switchers = As_all[:, mskdummies].sum(axis = -1)==1 # no need ot filter swithchers

		mask_ = np.array([1]* len(parties_all))
		mask_[test_id_wo_switchers] =0
		mask_pre = mask_.copy()


	# print(len(As_ids))

	# switchdummies = [list(As_ids).index(i) for i in true_switchers if i in As_ids]

	# switch_speech_bool = As_all[:, switchdummies].sum(axis = -1)==1
	# print("no author", (As_all.sum(axis = 1) == 0).sum())
	# print(As_all.shape)

	# # sum(switch_speech_bool),len(switch_speech_bool)

	# switch_id = np.arange(len(switch_speech_bool))[switch_speech_bool]
	# train_id_wo_switchers = np.setdiff1d(np.arange(len(As_all)), switch_id) # for model eval, exclude switchers, add them back for insights
	# assert (np.array(np.where(train_id_wo_switchers_0)[0]) == train_id_wo_switchers).all()
	masks_test = mask_pre[test_id_wo_switchers]
	ids_test = ids_all[test_id_wo_switchers]
	ds_test = ds_all[test_id_wo_switchers]
	As_test = As_all[test_id_wo_switchers]
	thetas_test = thetas_all[test_id_wo_switchers]
	parties_test = parties_all[test_id_wo_switchers]
	print('test shape', As_test.shape)



	ids, ids_val = ids_all[train_id_wo_switchers], ids_all
	ds, ds_val = ds_all[train_id_wo_switchers], ds_all
	As, As_val = As_all[train_id_wo_switchers], As_all
	thetas0, thetas_val = thetas_all[train_id_wo_switchers], thetas_all
	parties, parties_val = parties_all[train_id_wo_switchers], parties_all
	# idles, idles_val = idles_all[train_id_wo_switchers], idles_all[train_id_wo_switchers]
	masks, masks_val = mask_pre[train_id_wo_switchers], mask_pre.copy()
	print(As.shape, As_val.shape)
	
	print('shape before shuffle', ds.shape)




	####################################################################################################



	

	print('loaded data')
	# t_accs = []
	mutualinfos=[]
	s_accs = []
	p_accs = []
	Party_test_id = np.matmul(As_test.T, parties_test)>0
	val_valid = np.sum(As_val.T, axis = -1)!=0
	#### on the even round update T, with orthogonal constrainst over P
	### on the odd round, fix T, slightly push F to orthogonal
	for round in range(FLAGS.alter_rounds):
		print('round {}'.format(round))

		updateT = None
		do_val = True

		writesummary = True

		print('shuffle data')
		index = np.arange(len(ds))
		np.random.shuffle(index)
		ds = ds[index]
		As = As[index]
		# thetas = thetas[index]
		thetas0 = thetas0[index]
		parties = parties[index]
		ids = ids[index]
		masks = masks[index]
		print('shape after shuffle', ds.shape, len(index))
		# sort the input, non masked first



		trains = [ds, As, thetas0, parties, ids, masks]
		vals = [ds_val, As_val, thetas_val, parties_val, ids_val, masks_val]
		tests = [ds_test, As_test, thetas_test, parties_test, ids_test, masks_test]
		if do_val:
			speechids, _, _, _, val_Zmus, val_Frames, val_ids, val_parties, val_authors, Vampriors, Zp_mus, Zp_sigmas,val_slants, tst_Zmus, tst_slants= train(trains, vals, tests, Zbar = Zbar0, round = round, writesummary = writesummary, do_val = do_val)
			# print("\n########################vampriors and Zp_mu sigmas in val##################################\n", Vampriors, Zp_mus, Zp_sigmas)
			Vampriors  = np.array(Vampriors)
			Zp_mus  = np.array(Zp_mus)
			Zp_sigmas  = np.array(Zp_sigmas)
			

			np.save(FLAGS.resultdir + '/vampriors_{}'.format(round), Vampriors)
			np.save(FLAGS.resultdir + '/zp_mus_{}'.format(round), Zp_mus)
			np.save(FLAGS.resultdir + '/zp_sigmas_{}'.format(round), Zp_sigmas)

			# print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&check vamp shape!!!!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", Vampriors.shape, Zp_mus.shape, Zp_sigmas.shape)

			print('check doc parties prediction')
			s_acc =sum ((tst_slants>0) ==parties_test)/len(parties_test)
			print('slant accuracy is {} at round {} in test'.format(s_acc, round))
			s_accs.append(s_acc)


			print("check membership prediction")

			p_acc = predict_party_membership(tst_slants, Party_test_id, As_test)
			print('party membership accuracy is {} at round {} in test'.format(p_acc, round))
			p_accs.append(p_acc)

		# thetas_n = np.array(topics_mix)
		assert list(ids)==list(speechids), 'id order changed'

		Parties = np.array(val_parties)
		Frames = np.array(val_Frames)
		Zmus = np.array(val_Zmus) # collection of document z_mu

		Party_by_id= np.matmul(As_val.T, Parties)>0 # if 0 then from R or no speeches, will differenitiate




		Zbar = np.matmul(As_val.T, Zmus)/(np.sum(As_val, axis = 0)[:, np.newaxis]+0.000001)


		# Zbar change

		Zbar_diff = np.nanmean(np.linalg.norm(Zbar - Zbar0, axis = -1)/np.linalg.norm(Zbar0, axis = -1))

		print("Average Zbar changed by ", Zbar_diff)

		# update Zbar0 for the next round
		Zbar0 = Zbar

		# get MIs

		mutualinfo = PCA_mutual_info(Zbar, Party_by_id, As_val)
		# mutualinfo  = PCA_mutual_info(Zbar, Senate_by_id, As_val)
		print("Mutual information between Zbar clusters and parties D1/R0 is in validation:", mutualinfo)

		Zbar_t = np.matmul(As_test.T, tst_Zmus)/(np.sum(As_test, axis = 0)[:, np.newaxis]+0.000001)

		Party_test_id = np.matmul(As_test.T, parties_test)>0

		mutualinfo = PCA_mutual_info(Zbar_t, Party_test_id, As_test)
		print("Mutual information between Zbar clusters and parties D1/R0 is in test_size:", mutualinfo)
		mutualinfos.append(mutualinfo)







		# saving for future usages

		np.save(FLAGS.resultdir + '/speechids_{}'.format(round), val_ids) # to check if orders are wrong
		np.save(FLAGS.resultdir + '/Zmus_{}'.format(round), Zmus)
		np.save(FLAGS.resultdir + '/Frames_{}'.format(round), Frames) #edited by XR: save f
		np.save(FLAGS.resultdir + '/Zbars_{}'.format(round), Zbar[val_valid])
		np.save(FLAGS.resultdir + '/Slants_pred_{}'.format(round), val_slants)
	# np.save(FLAGS.resultdir + '/Frames_f_{}'.format(round), Frames)
	np.save(FLAGS.resultdir + '/mutual_info_scores', np.array(mutualinfos))

	return s_accs, p_accs, mutualinfos # all from the latest round


import shutil
if __name__ == '__main__':
	# non_iterative_training()
	if FLAGS.remove_history:
		try:
			shutil.rmtree(FLAGS.logdir)
		except:
			pass
		os.mkdir(FLAGS.logdir)
		# os.mkdir(FLAGS.logdir+ '/trial')
		try:
			shutil.rmtree(FLAGS.modelpath)
		except:
			pass
		os.mkdir(FLAGS.modelpath)
		try:
			shutil.rmtree(FLAGS.resultdir)
		except:
			pass
		os.mkdir(FLAGS.resultdir)
		# os.mkdir(FLAGS.modelpath +'/trial/')



	if FLAGS.tuning:
		# parameterdict = {"kl_anneal":[10,  15, 20, 25], "masks":[0.01,  0.02, 0.05, 0.1, 0.8], "std":[0.1, 0.2, 0.3, 0.4]}
		# parameterdict = {"kl_anneal":[10], "masks":[ 0.02], "std":[0.2, 0.3, 0.4]}
		# parameterdict = {"kl_anneal":[15, 18, 20], "masks":[ 0.02], "std":[0.3], 'epochs':[8, 10, 12]}
		parameterdict = {"masks":[  0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03, 0.01], 'epochs':[10]} # epochs >=5
		# parameterdict = {"masks":[ 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03],  'epochs':[8, 10, 15]}


		# parametergrid = [[i, j , k] for i in parameterdict["kl_anneal"] for j in parameterdict['beta_z'] for k in parameterdict['epochs'] if (k <i) and (k>=i/2)]
		# parametergrid = [[10, 2, 0.08], [10, 3, 0.1]]
		parametergrid = [[i, j] for i in parameterdict['masks'] for j in parameterdict['epochs']]
		print(parametergrid)
		all_MIs= []
		
		oldmodelp = FLAGS.modelpath
		oldresultp = FLAGS.resultdir
		oldlogp = FLAGS.logdir
		# print(parametergrid)
		with open(FLAGS.resultdir+'all_mutual_info_{}.txt'.format(vs), 'w') as fo:
			fo.write('train vampriors Zbar only {}, biased sample is {}\n'.format(vs, FLAGS.biased))

		for msk, epc in parametergrid:
			print(msk, epc)

			# FLAGS.anneal_kl = float(ann)
			FLAGS.notmask = msk
			# FLAGS.num_comp = int(com)
			FLAGS.n_epochs = epc
			versname = "msk_{}_epc_{}".format(msk, epc)



			FLAGS.modelpath = oldmodelp + versname+'/'
			FLAGS.logdir = oldlogp+ versname+'/'
			FLAGS.resultdir =oldresultp + versname+'/'
			try:


				os.mkdir(FLAGS.modelpath)
				os.mkdir(FLAGS.logdir)
				os.mkdir(FLAGS.resultdir)
			except:
				pass



			s_accs, p_accs, MIs = iterative_training()
			MIs = np.array(MIs)
			all_MIs.append(MIs)
			print(MIs)

			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			a = len(s_accs)
			ax1.plot(np.arange(a), p_accs, '.r-',  label='parties')
			# ax1.plot(np.arange(a), t_accs, )
			ax1.plot(np.arange(a), MIs, 'xb-', label='MutualInfos')
			ax1.plot(np.arange(a), s_accs, 'g-', label='doc slants')
			plt.legend(loc='upper left');
			ax1.set_title('training evaluation version {}'.format(vs))
			plt.savefig(FLAGS.resultdir + 'full sample evaluation_{}.jpg'.format(versname))
			plt.close()
			f = open(oldresultp+'all_mutual_info_{}.txt'.format(vs), 'a') 

			f.write(versname)
			f.write('\t')

			f.write(str(MIs))
			f.write('\t')
			f.write(str(list(s_accs)))	
			f.write('\t')
			f.write(str(list(p_accs)))
			f.write('\n')
			print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
			print(versname, MIs, s_accs, p_accs)
			print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
			f.close()


			# plt.show()


			# print(results)
			# exit()		
		# exit()



	elif FLAGS.iter:


		s_accs, p_accs,MIs = iterative_training()
		MIs = np.array(MIs)
		print(MIs)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		a = len(s_accs)

		# ax1.plot(np.arange(a), t_accs, )
		ax1.plot(np.arange(a), p_accs, '.r-',  label='parties')
		ax1.plot(np.arange(a), MIs, 'xb-', label='MutualInfos')
		ax1.plot(np.arange(a), s_accs, 'g-', label='doc slants')
		plt.legend(loc='upper left');
		ax1.set_title('training evaluation version {}'.format(vs))
		plt.savefig(FLAGS.modelpath + 'full sample evaluation.jpg')
		plt.show()
