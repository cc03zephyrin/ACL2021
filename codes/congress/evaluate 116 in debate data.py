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
vs = 116.0# vamprior, adding masked labels, randomly masked
penalty = 1
rp = 4
topic_dic = {0: 'ip', 1: 'it', 2: 'abortion', 3: 'academic', 4: 'agriculture', 5: 'business', 6: 'children', 7: 'china', 8: 'commerce', 9: 'crime', 
10: 'culture', 11: 'homelandsecurity', 12: 'detainee', 13: 'disadvantaged', 14: 'disaster', 15: 'discrimination', 16: 'disease', 
17: 'addictives', 18: 'economy', 19: 'education', 20: 'environment', 21: 'family', 22: 'federal', 23: 'finance', 24: 'fiscal', 
25: 'food', 26: 'gun', 27: 'health', 28: 'healthcare', 29: 'hightech', 30: 'housing', 31: 'immigration', 32: 'industry', 33: 'cyber',
34: 'infrastructure', 35: 'international', 36: 'iransyrialibya', 37: 'iraq', 38: 'israel', 39: 'jury', 40: 'lgbtq', 41: 'media',
42: 'militarycomplex', 43: 'natives', 44: 'nuclear', 45: 'police', 46: 'politicalact', 47: 'postal', 48: 'randd', 49: 'religion',
50: 'renewable', 51: 'reserves', 52: 'russia', 53: 'safety', 54: 'sport', 55: 'tax', 56: 'terrorism', 57: 'trade',
58: 'traditionalenergy', 59: 'transportation', 60: 'veteran', 61: 'vietnam', 62: 'vote', 63: 'waste', 64: 'welfare', 
65: 'woman', 66: 'workforce', 67:'other'}

topic_ids = [2, 26, 28, 40]


flags =tf.compat.v1.flags

flags.DEFINE_bool('biased', False, 'enabled biased sampling')
flags.DEFINE_bool('remove_history', True, 'remove training history logs and models')
flags.DEFINE_bool('iter', True, 'do iterative training')
flags.DEFINE_integer('round', 1, 'rounds') 

flags.DEFINE_bool('tuning', True, "tuning parameters")
flags.DEFINE_float('notmask', 0.05, 'non_masked label')

# flags.DEFINE_string('data_dir', r'C:\code\chen_proj_codes\news_bias\congressionalreport\processed\congress_report_topics\data_for_train', 'Directory for data')
# flags.DEFINE_float('beta_the', 1, 'coef of theta ')
flags.DEFINE_float('beta_kl', 1, 'coef for the kl diveregence of the priors')
flags.DEFINE_float('beta_th', 5.0, 'coef of theta constraints to original theta0 set to 1,') #  this is actual number / batchsize *2
flags.DEFINE_float('beta_t', 5.0, 'coef of constraints of T close to T0 or T00')
flags.DEFINE_float('beta_z', 15.0, 'z_mu stability, close to zbar')
flags.DEFINE_float('beta_p', 2, 'topic vector orthogonal to polarization')
flags.DEFINE_float('beta_s', 10, 'prediction of slants')
flags.DEFINE_float('beta_mse',250.0, 'coeff of mse, if standard normal set to 250')
flags.DEFINE_integer('alter_rounds', 1, 'alternate rounds') #  default 10
flags.DEFINE_integer('n_epochs', 5, 'number of epochs') # default 30
flags.DEFINE_float("pseudu_mean", 0.02, 'expectation of vamprior')# set to a value close to 0
flags.DEFINE_float('pseudu_std', 0.3, 'std of vamprior')
flags.DEFINE_integer('num_comp', 2, 'number of Gaussian mixtures, default 2')
flags.DEFINE_float('anneal_kl', 15, 'turn off regulization until KL reaches high value') #mean 6 large 13
flags.DEFINE_bool('Anneal', True, 'if true, turn off regulization until KL reaches high value')



# For bigger model:
flags.DEFINE_string('data_dir', r'/home/cczephyrin/projects/political embedding/training_vec/', 'Directory for data')
flags.DEFINE_string('logdir', r'/home/cczephyrin/projects/political embedding/logs/vae_semi_v{}/'.format(vs), 'Directory for logs')
flags.DEFINE_string('resultdir', r'/home/cczephyrin/projects/political embedding/results/validation/debate_train_v{}.{}/'.format(vs, rp), 'directory for results')
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

	print(" clustering attributes mean variance and weights for 1 components")
	print(gm.means_)
	print(gm.covariances_)
	print(gm.weights_)
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

	print(" clustering attributes mean variance and weights for 2 components")
	print(gm.means_)
	print(gm.covariances_)
	print(gm.weights_)
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

	print(" clustering attributes mean variance and weights")
	print(gm.means_)
	print(gm.covariances_)
	print(gm.weights_)
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


def model(TrainD, TrainA, TrainThe, TrainID, TrainParty, TrainMask,T0, T00, P0, Zbar = None, updateT = None, trained = True, reuse = False):
	trainf = True
	traint = True 


	with tf.name_scope('data'):
		# _id = tf.placeholder(tf.int64)
		# _party = tf.placeholder(tf.int64)
		# _d = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
		# _A = tf.placeholder(tf.float32, shape=(None, FLAGS.n_authors))

		P = tf.cast(tf.constant(P0), tf.float32)  # change to P0
		# _theta = tf.placeholder(tf.float32, shape = (None, FLAGS.n_topics))
		batchno = tf.Variable(0, dtype=tf.int64)
		TrainParty = tf.cast(TrainParty, tf.float32)
		if Zbar is not None:
			Zbar = tf.cast(tf.constant(Zbar), tf.float32) 
		else:
			Zbar = tf.constant(np.random.normal(size = (FLAGS.n_authors, FLAGS.latent_dim)))

		zbar = tf.gather(Zbar, tf.argmax(TrainA, -1))
		Mask = tf.cast(TrainMask, tf.float32)
		# # print('zbar shape!!!!',tf.shape(zbar))



		# TrainD = tf.cast(TrainD, tf.float32)					

	with tf.variable_scope('fullyconnected_t', reuse = reuse):
		# t_mu, t_sigma = topic_inference_network(TrainD, topic_dim=FLAGS.n_topics, hidden_size=FLAGS.hidden_size)
		t_z = topic_inference_network(TrainD, topic_dim=FLAGS.n_topics, hidden_size=FLAGS.hidden_size)
		# The variational distribution is a Normal with mean and standard
		# deviation given by the inference network
		# t_z = distributions.Normal(loc=t_mu, scale=t_sigma)
		# theta_n = t_z.sample()
		# tf.cast(theta_n, tf.float32)
		theta_n = tf.nn.softmax(t_z, axis=-1)



		# T = tf.Variable(T0, name = 'topic_matrix', dtype= tf.float32

		T = tf.get_variable("topic_matrix", dtype=tf.float32, initializer=tf.constant(T0))

			# biases = tf.Variable(tf.zeros([FLAGS.latent_dim]), trainable = False)
		# biases = tf.Variable(tf.zeros([FLAGS.latent_dim])) # edited by XR: match dims. todo: define a new flag for this
		biases = tf.Variable(tf.zeros([300]))
		# get topic vector for the specific toic
		t = tf.matmul(theta_n, T) + biases
		P_ = tf.nn.l2_normalize(P, axis = -1)
		T_ = tf.nn.l2_normalize(T, axis = -1)
		p_t= tf.reduce_sum(tf.abs(tf.reduce_sum(P_ * T_, axis = -1)), axis = -1) # inner product of T and P for the same topic, weighted by theta_n


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
		z_theta = tf.concat((z_sample, theta_n), axis=-1)
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
		f = framing_inference_network_decoder(z_theta, hidden_size=FLAGS.hidden_size, latent_output_dim= FLAGS.latent_output_dim) ### Chen: use flag for consistency
		d_recon = f
		p_x_given_z = d_recon + t

	# Generate vampriors, of size C * inputdim
	with tf.variable_scope('vamprior', reuse = reuse):
		vamps = gen_vampriors(FLAGS.num_comp, FLAGS.latent_output_dim) #C * 300
		vamps = tf.cast(vamps, tf.float32)
		tf.summary.scalar('mean of vamps', tf.reduce_mean(vamps))
		# print('vamps  shape ', tf.shape(vamps))
		# vamp = tf.reduce_mean(vamps)

	with tf.variable_scope('variational_f', reuse = True):

		z_p_mu, z_p_logvar = framing_inference_network(TrainA, TrainThe, vamps, latent_dim=FLAGS.latent_dim,
												  hidden_size=FLAGS.hidden_size)
		# z_p_var = tf.pow(z_p_sigma, 2)
		# z_p_logvar = tf.math.log(z_p_var


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

		mse = FLAGS.beta_mse* tf.losses.mean_squared_error(TrainD, p_x_given_z)/2.0  # assuming the variance is 1
		zmuLoss = FLAGS.beta_z * tf.losses.mean_squared_error(z_mu, zbar)

		SlantLoss = FLAGS.beta_s * Mask * custom_crossentrophy(s_logit, TrainParty)

		ThetaLoss = FLAGS.beta_th *tf.keras.losses.kullback_leibler_divergence(tf.to_float(TrainThe), theta_n)
		TLoss =  FLAGS.beta_p * p_t 
		TmatrixLoss = FLAGS.beta_t * tf.nn.l2_loss(T - T00) # use T0 if we want to iteratively update T, T00 is the origianl T

        # below edited by XR, do not need to predict s: 
		# FLoss = FLAGS.beta_f * tf.losses.sigmoid_cross_entropy(multi_class_labels=TrainParty, logits=s_logit) # sahre the same cooef with T update
		if updateT is None:
			# elbo = tf.reduce_mean(- mse  - 0.2 * kl_ft  - ThetaLoss - TmatrixLoss - FLoss - TLoss * 0.2 - FFtLoss, 0)  # 150 is 300/2
			elbo = tf.reduce_mean(- mse -kl_mu  - ThetaLoss - TmatrixLoss  -SlantLoss, 0)
			# elbo = tf.reduce_mean(-mse - kl_mu - ThetaLoss - TmatrixLoss - SlantLoss)
			# elbo = tf.reduce_mean(- mse - kl_f  - ThetaLoss - TmatrixLoss - FLoss - TLoss , 0)


		optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

		train_op = optimizer.minimize(-elbo)
		# if writesummary:

		train_ops = [train_op,  batchno.assign_add(1)] # edited by XR: delete f, change ft to f


		return train_ops

	else:

		# summary_op = tf.summary.merge_all()
		# val_ops = [summary_op, TrainID, TrainParty, theta_n, TrainThe, ft, TrainA, mse, s_pred, batchno.assign_add(1)] 
		val_ops = [ TrainID,theta_n,  s_logit, batchno.assign_add(1)] # edited by XR: change ft to f

	
		return val_ops
# def model(TrainD, TrainA, TrainThe, TrainID, TrainParty, TrainMask,T0, T00, P0, Zbar = None, updateT = None, trained = True, reuse = False):
	trainf = True
	traint = True 


	with tf.name_scope('data'):
		# _id = tf.placeholder(tf.int64)
		# _party = tf.placeholder(tf.int64)
		# _d = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
		# _A = tf.placeholder(tf.float32, shape=(None, FLAGS.n_authors))

		P = tf.cast(tf.constant(P0), tf.float32)  # change to P0
		# _theta = tf.placeholder(tf.float32, shape = (None, FLAGS.n_topics))
		batchno = tf.Variable(0, dtype=tf.int64)
		TrainParty = tf.cast(TrainParty, tf.float32)
		if Zbar is not None:
			Zbar = tf.cast(tf.constant(Zbar), tf.float32) 
		else:
			Zbar = tf.constant(np.random.normal(size = (FLAGS.n_authors, FLAGS.latent_dim).astype(np.float32)))

		zbar = tf.gather(Zbar, tf.argmax(TrainA, -1))
		Mask = tf.cast(TrainMask, tf.float32)
		# print('zbar shape!!!!',tf.shape(zbar))



		# TrainD = tf.cast(TrainD, tf.float32)					

	with tf.variable_scope('fullyconnected_t', reuse = reuse):
		# t_mu, t_sigma = topic_inference_network(TrainD, topic_dim=FLAGS.n_topics, hidden_size=FLAGS.hidden_size)
		t_z = topic_inference_network(TrainD, topic_dim=FLAGS.n_topics, hidden_size=FLAGS.hidden_size)
		# The variational distribution is a Normal with mean and standard
		# deviation given by the inference network
		# t_z = distributions.Normal(loc=t_mu, scale=t_sigma)
		# theta_n = t_z.sample()
		# tf.cast(theta_n, tf.float32)
		theta_n = tf.nn.softmax(t_z, axis=-1)



		# T = tf.Variable(T0, name = 'topic_matrix', dtype= tf.float32

		T = tf.get_variable("topic_matrix", dtype=tf.float32, initializer=tf.constant(T0))

			# biases = tf.Variable(tf.zeros([FLAGS.latent_dim]), trainable = False)
		# biases = tf.Variable(tf.zeros([FLAGS.latent_dim])) # edited by XR: match dims. todo: define a new flag for this
		biases = tf.Variable(tf.zeros([300]))
		# get topic vector for the specific toic
		t = tf.matmul(theta_n, T) + biases
		P_ = tf.nn.l2_normalize(P, axis = -1)
		T_ = tf.nn.l2_normalize(T, axis = -1)
		p_t= tf.reduce_sum(tf.abs(tf.reduce_sum(P_ * T_, axis = -1)), axis = -1) # inner product of T and P for the same topic, weighted by theta_n


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
		z_theta = tf.concat((z_sample, theta_n), axis=-1)
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
		f = framing_inference_network_decoder(z_theta, hidden_size=FLAGS.hidden_size, latent_output_dim= FLAGS.latent_output_dim) ### Chen: use flag for consistency
		d_recon = f
		p_x_given_z = d_recon + t

	# Generate vampriors, of size C * inputdim
	with tf.variable_scope('vamprior', reuse = reuse):
		vamps = gen_vampriors(FLAGS.num_comp, FLAGS.latent_output_dim) #C * 300
		vamps = tf.cast(vamps, tf.float32)
		tf.summary.scalar('mean of vamps', tf.reduce_mean(vamps))
		# print('vamps  shape ', tf.shape(vamps))
		# vamp = tf.reduce_mean(vamps)

	with tf.variable_scope('variational_f', reuse = True):

		z_p_mu, z_p_logvar = framing_inference_network(TrainA, TrainThe, vamps, latent_dim=FLAGS.latent_dim,
												  hidden_size=FLAGS.hidden_size)
		# z_p_var = tf.pow(z_p_sigma, 2)
		# z_p_logvar = tf.math.log(z_p_var


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

		mse = FLAGS.beta_mse* tf.losses.mean_squared_error(TrainD, p_x_given_z)/2.0  # assuming the variance is 1
		zmuLoss = FLAGS.beta_z * tf.losses.mean_squared_error(z_mu, zbar)
		SlantLoss = FLAGS.beta_s * Mask * custom_crossentrophy(s_logit, TrainParty)

		ThetaLoss = FLAGS.beta_th *tf.keras.losses.kullback_leibler_divergence(tf.to_float(TrainThe), theta_n)
		TLoss =  FLAGS.beta_p * p_t 
		TmatrixLoss = FLAGS.beta_t * tf.nn.l2_loss(T - T00) # use T0 if we want to iteratively update T, T00 is the origianl T

        # below edited by XR, do not need to predict s: 
		# FLoss = FLAGS.beta_f * tf.losses.sigmoid_cross_entropy(multi_class_labels=TrainParty, logits=s_logit) # sahre the same cooef with T update
		if updateT is None:
			# elbo = tf.reduce_mean(- mse  - 0.2 * kl_ft  - ThetaLoss - TmatrixLoss - FLoss - TLoss * 0.2 - FFtLoss, 0)  # 150 is 300/2
			elbo = tf.reduce_mean(- mse -kl_mu  - ThetaLoss - TmatrixLoss - zmuLoss -SlantLoss, 0)
			# elbo = tf.reduce_mean(-mse - kl_mu - ThetaLoss - TmatrixLoss - SlantLoss)
			# elbo = tf.reduce_mean(- mse - kl_f  - ThetaLoss - TmatrixLoss - FLoss - TLoss , 0)


		optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

		train_op = optimizer.minimize(-elbo)
		# if writesummary:
		elbo_loss = tf.summary.scalar('elbo_m', tf.reduce_mean(elbo))
		mse_loss = tf.summary.scalar('mse_m', tf.reduce_mean(mse))
		kl_mu_loss = tf.summary.scalar('kl_mu_gmmvampriors', tf.reduce_mean(kl_mu))
		# kl_t_loss = tf.summary.scalar('kl_t_m', tf.reduce_mean(kl_t))
		Theta_loss = tf.summary.scalar('theta loss', tf.reduce_mean(ThetaLoss))
		T_loss = tf.summary.scalar('T oppose to polar loss', tf.reduce_mean(TLoss))
		zmu_loss = tf.summary.scalar('zmu oppose to zbar loss', tf.reduce_mean(zmuLoss))
		z_v_prior = tf.summary.scalar('z prob given pseudo input', tf.reduce_mean(p_z_given_v))
		z_x_prior = tf.summary.scalar('z prob given actual input', tf.reduce_mean(p_z_given_x))
		# vamp_check = tf.summary.scalar('check vampriors', tf.reduce_mean(vamps))

        #below edited by XR
		S_loss = tf.summary.scalar('Predicted slant oppose to party slant loss', tf.reduce_mean(SlantLoss))
		# Fft_loss = tf.summary.scalar('F oppose to Ft', tf.reduce_mean(FFtLoss))

		Tmatrix_loss = tf.summary.scalar('T matrix loss', tf.reduce_mean(TmatrixLoss))
		for v in tf.trainable_variables():
			tf.summary.scalar(v.name, tf.reduce_mean(v))

		# sigma_loss = tf.summary.scalar('topic sigma losss', tf.reduce_mean(sigmaLoss))
		# summary_op = tf.summary.merge([elbo_loss, mse_loss, kl_ft_loss, Theta_loss, T_loss, F_loss, Tmatrix_loss, Fft_loss]) # edited by XR
		summary_op = tf.summary.merge_all()
		# summary_ops = [summary_op, elbo, mse, kl_ft, train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,
		#			   batchno.assign_add(1)] #edited by XR: change f to z, ft to f
		summary_ops = [summary_op, elbo, mse, kl_mu, train_op, TrainID, TrainParty, T, theta_n, TrainThe, z, f,vamps,  batchno.assign_add(1)]
			# print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)

		# Merge all the summaries

		# train_ops = [train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,batchno.assign_add(1)]
		train_ops = [train_op, TrainID, TrainParty, T, theta_n, TrainThe, z, f, vamps,  kl_mu, batchno.assign_add(1)] # edited by XR: delete f, change ft to f


		return train_ops, summary_ops, 

	else:
		mse = FLAGS.beta_mse* tf.losses.mean_squared_error(TrainD, p_x_given_z)/2.0
		# sigmaLoss = FLAGS.beta_f*tf.nn.l2_loss(sigma)  # optional, 
		# FLoss = FLAGS.beta_f * tf.abs(tf.matmul(f, tf.transpose(t)))
		# FLoss = FLAGS.beta_f * tf.losses.sigmoid_cross_entropy(multi_class_labels=TrainParty, logits=s_logit) # edited by XR, do not predict s
		TLoss = FLAGS.beta_p * p_t
		zmuLoss = FLAGS.beta_z * tf.losses.mean_squared_error(z_mu, zbar)
		# ThetaLoss = FLAGS.beta_th *  tf.losses.mean_squared_error(
		# 	tf.to_float(TrainThe), theta_n)
		ThetaLoss = FLAGS.beta_th * tf.nn.l2_loss(tf.to_float(TrainThe)-
			theta_n)
		# F_loss_val = tf.summary.scalar('F oppose to slant loss _val', tf.reduce_mean(FLoss)) # edited by XR
		Theta_loss_val = tf.summary.scalar('theta loss_val', tf.reduce_mean(ThetaLoss))
		T_loss_val = tf.summary.scalar('T oppose to polar loss _val', tf.reduce_mean(TLoss))
		zmu_loss_val = tf.summary.scalar('zmu oppose to zbar loss in val, some zbar will be 0', tf.reduce_mean(zmuLoss))
		mse_loss_val = tf.summary.scalar('mse_val', tf.reduce_mean(mse))
		summary_op = tf.summary.merge([mse_loss_val, zmu_loss_val, Theta_loss_val, T_loss_val]) #edited by XR
		# summary_op = tf.summary.merge_all()
		# val_ops = [summary_op, TrainID, TrainParty, theta_n, TrainThe, ft, TrainA, mse, s_pred, batchno.assign_add(1)] 
		val_ops = [summary_op, TrainID, TrainParty, theta_n, TrainThe, z,f, vamps, z_p_mu, z_p_logvar, TrainA, mse, s_logit, batchno.assign_add(1)] # edited by XR: change ft to f

	
		return val_ops


def train(trains, tests, T0, T00,P0 = np.zeros([FLAGS.n_topics, FLAGS.latent_dim]), Zbar = None,  updateT = None, round = 0, do_val = False, writesummary = False):
	# Train a Variational Autoencoder				batchcount = 0

	# for each round, update thetas, and P0, T0

	# Input placeholders

	Ds, As, Thetas, Parties, Ids, Masks = trains

	Ds_test, As_test, Thetas_test, Parties_test, Ids_test, Masks_test= tests
	print(Ds.shape)
	print(As.shape)

	print(Thetas.shape)
	Ds = np.float32(Ds)
	# Ds_val = np.float32(Ds_val)
	Ds_test = np.float32(Ds_test)

	C = FLAGS.num_comp
	round = FLAGS.round
	model_dir = FLAGS.modelpath
	print(model_dir)
	model_name = '/msk_0.05_epc_8/{}/vae_{}'.format(round, round)
	model_path = model_dir + model_name
	print(model_path)
	tf.reset_default_graph()


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


		test_dataset = tf.data.Dataset.from_tensor_slices((Ds_test, As_test, Thetas_test,  Ids_test, Parties_test, Masks_test))
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		TstD, TstA, TstThe, TstID, TstParty, TstMask= test_iterator.get_next()
		# tf.cast(TrainD, tf.float32)
		# tf.cast(TrainA, tf.float32)
		

		train_ops= model(TrainD, TrainA, TrainThe, TrainID, TrainParty, TrainMask,T0, T00,P0, Zbar = Zbar, updateT = updateT)
		saver = tf.train.Saver()
		test_ops = model(TstD, TstA, TstThe, TstID, TstParty, TstMask,T0, T00,P0, trained = False, reuse = True, updateT = None, Zbar = Zbar)

		all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
		# print(all_variables)
		
		init_op = tf.global_variables_initializer()
		
		# with tf.InteractiveSession() as sess:
		with tf.Session() as sess:
			sess.run(init_op)
			saver.restore(sess = sess, save_path = model_path)

			print('start training')
			for epoch in range(FLAGS.n_epochs):
				print("start epoch {}".format(epoch))

				sess.run(train_iterator.initializer, feed_dict = {Dt:Ds, At:As, Thet:Thetas, Ptr:Parties, Idt:Ids, Maskt: Masks})
				# print(sess.run([TrainD, TrainParty, TrainThe]))


				batchcount = 0
				# vamps_= []
				while True:
					try:

						_, batchcount = sess.run(train_ops)  # Ds, As, Thetas, Parties,Ids


					except tf.errors.OutOfRangeError:

						break

			# start test
			print("begin test")
			sess.run(test_iterator.initializer)
			tst_ids = []
			tst_thetas = []
			# tst_parties = []
			tst_slants = []
			while True:
				try:
					TstID_,  theta_t_, S_pred_,batchcount = sess.run(test_ops) 
					tst_ids.extend(TstID_)
					tst_slants.extend(S_pred_)
					# tst_Zmus.extend(z_)
					tst_thetas.extend(theta_t_)
					# tst_parties.extend(TstParty_)
				except tf.errors.OutOfRangeError:
					break
			tst_ids = np.array(tst_ids)
			tst_slants = np.array(tst_slants)

			tst_thetas = np.array(tst_thetas)
			assert (tst_ids==Ids_test).all(), 'test ids no match, check id order to make sure order is not changed'

	

	if do_val:
		


		# Zmus = np.array(Zmus)
		# all zp_mus in val will be the sames 
		return  tst_slants,tst_thetas #edited by XR: save Frames later
	else:
		speechids = np.array(speechids)
		# Zmus = np.array(Zmus)
		return Tmatrix, topics_mix, speechids, Zmus, Frames, parties



def iterative_training():
	# load data, initialize P0 and T0

	datapath = FLAGS.data_dir + 'debabe_test/'
	ds_all = np.load(os.path.join(datapath, 'debate_sentencevector.npy'))
	documents = pd.read_csv(r'/home/cczephyrin/projects/political embedding/data/somas_debate_4topics_processfiltered.csv', sep = '\t', encoding = 'ISO-8859-1')
	print(documents.shape)
	textraw = documents['body'].values
	print(textraw[:2])

	enddate = '2020-10-04'
	# retain gun topic only 
	thetas_soft_all = np.load(os.path.join(datapath, 'soft_topic.npy'))
	
	thetas_hard_all = np.load(os.path.join(datapath, 'hard_topic.npy'))
	theta_categ_all = np.argmax(thetas_hard_all, axis = -1)
	# inx = (theta_categ_all==26)|(theta_categ_all==40)
	inx = (theta_categ_all==26)
	theta_categ_ = theta_categ_all[inx]
	# ds_all = np.load(os.path.join(datapath, 'cong_reports_102_114_for_training_sif.npz'))['arr_0'][bottom:]

	# oversample = SMOTE()
	
	parties_pre = np.load(os.path.join(datapath, 'stance_lib.npy'))[inx]
	y = parties_pre.copy()
	indx =np.where(y>0)

	newX = np.random.choice(indx[0], 350)
	print(newX)
	X_ = np.arange(len(y))
	X = np.concatenate([X_, newX])


	
	# print(X)

	# print(len(ds_all))
	parties_all = parties_pre[X]

	print(parties_all.mean())
	# print(parties_pre)
	# exit()
	Zmus_pre = np.load(os.path.join(FLAGS.data_dir, 'Z_mus_pre.npy'))[:4227][inx][X]
	ds_all = np.float32(ds_all)[inx][X]
	thetas_all = thetas_hard_all[inx][X]
	theta_categ = theta_categ_[X]

	# print(ds.shape)
	As_all = np.load(os.path.join(datapath, 'fake_author.npy'))[inx][X]

	print("Author matrix shape is ", As_all.shape)

	T00 = np.load(os.path.join(FLAGS.data_dir, 'debate_Topicvectors_68dim.npy'))

	ids_all = np.load(os.path.join(datapath, 'speech_ids.npy'), allow_pickle= True)[inx][X]
	P0 = np.load(os.path.join(FLAGS.data_dir, 'PR_axis_piror_1991_{}.npy'.format(enddate)))
	P0 = np.roll(P0, -1, axis = 0) # orignal P0 first row corresponding to the null topic, which is the last topic,
	# P0 = np.load(r'C:\code\chen_proj_codes\\news_bias\congressionalreport\processed\congress_report_topics\trial_data\PR_sample_piror.npy')
	assert P0.shape ==(68, 300), 'dr axis prior shape error'

	# thetas_all = thetas0_all.copy()
	T0 = T00.copy()
	# print('loaded data')
	# split

	# calculate Zbar0 matrix

	Zbar0 = np.matmul(As_all.T, Zmus_pre)/(np.sum(As_all, axis = 0)[:, np.newaxis]+0.000001)
	# print("dim of Zbar0", Zbar0.shape, Zbar0.sum(axis =-1))
	# print('Zbar0', Zbar0[-5:])
	# print("number of talks", As_all.T.sum(axis = 1)[-5:])

	# # generate idle inputs for training

	# i_all = np.array([np.eye(FLAGS.num_comp, dtype = np.float32)] * len(ds_all))
	# print('idles shape', idles_all.shape)
	# generate mask
	mask_ = np.array([0]* len(parties_all))
	if not FLAGS.biased:
		sss = StratifiedShuffleSplit(n_splits=5, test_size= 1- FLAGS.notmask)
		train_id, test_id = next(sss.split(theta_categ, parties_all))
		mask_[train_id] = 1
		# print("non zero results", sum(mask_))
		mask_pre = mask_.copy()
	else:
		dws_all = np.float32(np.load(os.path.join(datapath, 'cr_authordw_bydoc_1991_{}.npy'.format(enddate))))[inx]

		orderid = np.argsort(dws_all[:, 0])
		tbn = int(len(parties_all) * FLAGS.notmask/2)
		train_id = np.concatenate([orderid[:tbn], orderid[-tbn:]])
		test_id = [i for i in range(len(orderid)) if i not in train_id]
		mask_[train_id] = 1
		mask_pre = mask_.copy()



	masks_test = mask_[test_id]
	ids_test = ids_all[test_id]
	ds_test = ds_all[test_id]
	As_test = As_all[test_id]
	thetas_test = thetas_all[test_id]
	parties_test = parties_all[test_id]
	thetacate_test = theta_categ[test_id]

	# sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
	# train_id, test_id = next(sss.split(As_all, parties_all))


	# train_id_wo_switchers = np.setdiff1d(train_id, switch_id)
	#test_id_wo_switchers = np.union1d(test_id, switch_id)

	# test_id_wo_switchers = np.setdiff1d(test_id, switch_id)

	# print(sum(switch_speech_bool),len(switch_speech_bool))
	# assert len(np.union1d(train_id_wo_switchers, test_id_wo_switchers)) == len(switch_speech_bool), 'delete wrong number of swithers'

	# party_id_ = np.matmul(As_all.T, parties_all)
	# party_id = np.where(party_id_ > 1, 1, 0)

	# for unsupervised learning, using all for training and all for validation, thus no test ids


	print('shape before shuffle', ds_all.shape)




	####################################################################################################



	

	print('loaded data')
	t_accs = []
	# mutualinfos=[]
	s_accs = []


	for round in range(FLAGS.alter_rounds):
		print('round {}'.format(round))

		updateT = None
		do_val = True


		print('shuffle data')
		index = np.arange(len(ds_all))
		np.random.shuffle(index)
		ds = ds_all[index]
		As = As_all[index]
		# thetas = thetas[index]
		thetas0 = thetas_all[index]
		parties = parties_all[index]
		ids = ids_all[index]
		masks = mask_pre[index]
		print('shape after shuffle', ds.shape, len(index))
		# sort the input, non masked first



		trains = [ds, As, thetas0, parties, ids, masks]
	
		tests = [ds_test, As_test, thetas_test, parties_test, ids_test, masks_test]
		if do_val:
			tst_slants,  tst_thetas= train(trains, tests, T0= T0, T00=T00, P0 = P0, Zbar= Zbar0, round = round, updateT= updateT, writesummary = False, do_val = do_val)


			print('check test topic prediction')
			topics_0_val = np.argmax(thetas_test, axis=1)
			topics_1_val = np.argmax(tst_thetas, axis=1)
			t_acc = sum(topics_0_val == topics_1_val) / len(topics_1_val)
			print('topic absolute accuracy is {} at round {} in test'.format(t_acc, round))

			print('check doc parties prediction')
			right = ((tst_slants>0) ==parties_test)
			s_acc =sum ((tst_slants>0) ==parties_test)/len(parties_test)
			print('slant accuracy is {} at round {} in test'.format(s_acc, round))
			t_accs.append(t_acc)
			s_accs.append(s_acc)
			print((tst_slants>0))

			text_misclass = textraw[test_id][~right]
			parties_misclass = parties_test[~right]
			topics_misclass = theta_categ[test_id][~right]
			with open(FLAGS.resultdir + 'misclassified.txt', 'w', encoding = 'ISO-8859-1') as f:
				for i, t in enumerate(text_misclass):
					f.write(str(parties_misclass[i]) + '\t')
					f.write(topic_dic[topics_misclass[i]] + '\t')
					f.write(t)
					f.write('\n')


	return   t_accs, s_accs


import shutil
if __name__ == '__main__':
	# non_iterative_training()
	if FLAGS.remove_history:
		try:
			shutil.rmtree(FLAGS.resultdir)
		except:
			pass
		os.mkdir(FLAGS.resultdir)
		# os.mkdir(FLAGS.modelpath +'/trial/')



	if FLAGS.tuning:
		# parameterdict = {"kl_anneal":[10,  15, 20, 25], "masks":[0.01,  0.02, 0.05, 0.1, 0.8], "std":[0.1, 0.2, 0.3, 0.4]}
		# parameterdict = {"kl_anneal":[10], "masks":[ 0.02], "std":[0.2, 0.3, 0.4]}
		# parameterdict = {"kl_anneal":[10,  15], "masks":[0.05, 0.03,0.02], "std":[0.3], 'epochs':[5, 8, 10]} # epochs >=5

		# parameterdict = {"masks":[ 0.8, 0.6, 0.4, 0.2,  0.1, 0.08, 0.05, 0.03, 0.01],  'epochs':[10]} # epochs >=5
		# parameterdict = {"masks":[ 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.03],  'epochs':[8, 10, 15]}
		parameterdict = {"masks":[0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03, 0.01],  'epochs':[20]} # epochs >=5

		# parameterdict = {"masks":[0.6],  'epochs':[15]}
		# parametergrid = [[i, j , k] for i in parameterdict["kl_anneal"] for j in parameterdict['beta_z'] for k in parameterdict['epochs'] if (k <i) and (k>=i/2)]
		# parametergrid = [[10, 2, 0.08], [10, 3, 0.1]]
		parametergrid = [[i, j] for i in parameterdict['masks'] for j in parameterdict['epochs']]
		print(parametergrid)
		all_MIs= []
		
		# oldmodelp = FLAGS.modelpath
		oldresultp = FLAGS.resultdir
		# oldlogp = FLAGS.logdir
		# print(parametergrid)
		with open(FLAGS.resultdir+'all_accuracy_{}.txt'.format(vs), 'w') as fo:
			fo.write('vs 116 {}, biased sample is {}\n'.format(vs, FLAGS.biased))

		for msk, epc in parametergrid:
			print(msk, epc)

			# FLAGS.anneal_kl = float(ann)
			FLAGS.notmask = msk
			# FLAGS.num_comp = int(com)
			FLAGS.n_epochs = epc
			versname = "msk_{}_epc_{}".format(msk, epc)



			t_accs, s_accs = iterative_training()
	


			f = open(oldresultp+'all_mutual_info_{}.txt'.format(vs), 'a') 

			f.write(versname)
			f.write('\t')

			f.write(str(s_accs[0]))
			f.write('\t')
			f.write(str(t_accs[0]))
			f.write('\n')
			print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
			print(versname, s_accs, t_accs)
			print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
			f.close()


			# plt.show()


			# print(results)
			# exit()		
		# exit()



	elif FLAGS.iter:


		t_accs, s_accs, p_accs, MIs = iterative_training()
		MIs = np.array(MIs)
		print(MIs)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		a = len(t_accs)
		ax1.plot(np.arange(a), p_accs, '.r-',  label='parties')
		# ax1.plot(np.arange(a), t_accs, )
		ax1.plot(np.arange(a), MIs, 'xb-', label='MutualInfos')
		ax1.plot(np.arange(a), s_accs, 'g-', label='doc slants')
		plt.legend(loc='upper left');
		ax1.set_title('training evaluation version {}'.format(vs))
		plt.savefig(FLAGS.modelpath + 'full sample evaluation.jpg')
		plt.show()
