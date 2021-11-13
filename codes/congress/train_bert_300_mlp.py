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
import warnings

# def fxn():
# 	warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
# 	warnings.simplefilter("ignore")
# 	fxn()

# from imageio import imwrite
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# sns.set_style('whitegrid')

# distributions = tf.distributions
vs = 300.12# vamprior, adding masked labels, randomly masked
penalty = 1


flags =tf.compat.v1.flags
flags.DEFINE_bool('biased', True, 'enabled biased sampling')
flags.DEFINE_bool('remove_history', True, 'remove training history logs and models')
flags.DEFINE_bool('iter', True, 'do iterative training')

flags.DEFINE_bool('tuning', True, "tuning parameters")
flags.DEFINE_float('notmask', 0.8, 'non_masked label')

# flags.DEFINE_string('data_dir', r'C:\code\chen_proj_codes\news_bias\congressionalreport\processed\congress_report_topics\data_for_train', 'Directory for data')
# flags.DEFINE_float('beta_the', 1, 'coef of theta ')
flags.DEFINE_float('beta_kl', 1, 'coef for the kl diveregence of the priors')
flags.DEFINE_float('beta_th', 5.0, 'coef of theta constraints to original theta0 set to 1,') #  this is actual number / batchsize *2
flags.DEFINE_float('beta_t', 5.0, 'coef of constraints of T close to T0 or T00')
flags.DEFINE_float('beta_z', 15.0, 'z_mu stability, close to zbar')
flags.DEFINE_float('beta_p', 2, 'topic vector orthogonal to polarization')
flags.DEFINE_float('beta_s', 10, 'prediction of slants')
flags.DEFINE_float('beta_mse',250.0, 'coeff of mse, if standard normal set to 250')
flags.DEFINE_integer('alter_rounds', 6, 'alternate rounds') #  default 10
flags.DEFINE_integer('n_epochs', 8, 'number of epochs') # default 30
flags.DEFINE_float("pseudu_mean", 0.02, 'expectation of vamprior')# set to a value close to 0
flags.DEFINE_float('pseudu_std', 0.3, 'std of vamprior')
flags.DEFINE_integer('num_comp', 2, 'number of Gaussian mixtures, default 2')
flags.DEFINE_float('anneal_kl', 20, 'turn off regulization until KL reaches high value') #mean 6 large 13
flags.DEFINE_bool('Anneal', True, 'if true, turn off regulization until KL reaches high value')



# For bigger model:
flags.DEFINE_string('data_dir', r'D:\projects\congress\processed/training_vectors/', 'Directory for data')
flags.DEFINE_string('logdir', r'D:\projects\congress\logs/vae_semi_v{}/'.format(vs), 'Directory for logs')
flags.DEFINE_string('resultdir', r'D:\projects\congress/results/framings/framing_train_v{}/'.format(vs), 'directory for results')
flags.DEFINE_string('modelpath', r'D:\projects\congress/models/vae_semi_v{}/'.format(vs), 'Directory for models')
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

	print(" clustering attributes mean variance and weights for one components")
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

	print(" clustering attributes mean variance and weights for two components")
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


def framing_inference_network(d, latent_dim, hidden_size,trainable = True):
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
	with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, trainable = trainable, weights_regularizer=slim.l2_regularizer(1e-8)):
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


	return mu





def slant_network(z, hidden_size, output_dim = 1):
	"""Build a generative network parametrizing the likelihood of the data
	Args:
	z: Samples of latent variables
	hidden_size: Size of the hidden state of the neural net
	Returns:
	bernoulli_logits: logits for the Bernoulli likelihood of the data
	"""
	with slim.arg_scope([slim.fully_connected], activation_fn= tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-8)):
		z = tf.to_float(z)
		net = slim.fully_connected(z, hidden_size)
		net = slim.fully_connected(net, hidden_size)
		output = slim.fully_connected(net, output_dim, activation_fn = None)
	# bernoulli_logits = tf.reshape(bernoulli_logits, [-1, 28, 28, 1])
	return output


def model(TrainD, TrainParty,  TrainID, trained = True, reuse = False):
	trainf = True
	traint = True 


	with tf.name_scope('data'):
		# _id = tf.placeholder(tf.int64)
		# _party = tf.placeholder(tf.int64)
		# _d = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
		# _A = tf.placeholder(tf.float32, shape=(None, FLAGS.n_authors))

		# P = tf.cast(tf.constant(P0), tf.float32)  # change to P0
		# _theta = tf.placeholder(tf.float32, shape = (None, FLAGS.n_topics))
		batchno = tf.Variable(0, dtype=tf.int64)
		TrainParty = tf.cast(TrainParty, tf.float32)
				



	with tf.variable_scope('variational_f', reuse = reuse):
		# if test:
		# 	d= TrainDr
		# else:
		# 	d = TrainD
		z_mu= framing_inference_network(TrainD, latent_dim=FLAGS.latent_dim,
												  hidden_size=FLAGS.hidden_size)


		


	with tf.variable_scope('slant_pred', reuse = reuse):
		
		s_logit = slant_network(z_mu, FLAGS.hidden_size, output_dim = 1)
		s_logit = tf.squeeze(s_logit, axis = 1)
		s_pred = tf.cast(s_logit > 0, tf.int64)



	if trained:

		SlantLoss = tf.losses.sigmoid_cross_entropy(multi_class_labels=TrainParty, logits=s_logit)# 

		# print("slant loss shape", TrainParty.shape, s_logit.shape)
		# try:
		# elbo = tf.reduce_mean(-SlantLoss, 0)

		# except:
		# 	return SlantLoss

		


		optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

		train_op = optimizer.minimize(SlantLoss)

		S_loss = tf.summary.scalar('Predicted slant oppose to party slant loss', tf.reduce_mean(SlantLoss))

		# sigma_loss = tf.summary.scalar('topic sigma losss', tf.reduce_mean(sigmaLoss))
		# summary_op = tf.summary.merge([elbo_loss, mse_loss, kl_ft_loss, Theta_loss, T_loss, F_loss, Tmatrix_loss, Fft_loss]) # edited by XR
		summary_op = tf.summary.merge([S_loss])
		# summary_ops = [summary_op, elbo, mse, kl_ft, train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,
		#			   batchno.assign_add(1)] #edited by XR: change f to z, ft to f
		summary_ops = [summary_op, train_op,   batchno.assign_add(1)]
			# print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)

		# Merge all the summaries

		# train_ops = [train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,batchno.assign_add(1)]
		train_ops = [train_op,  s_logit,TrainParty, SlantLoss, batchno.assign_add(1)] # edited by XR: delete f, change ft to f


		return train_ops, summary_ops

	else:
	
		SlantLoss = tf.losses.sigmoid_cross_entropy(multi_class_labels=TrainParty, logits=s_logit)# 
	
		val_ops = [SlantLoss,TrainID, TrainParty, z_mu, s_logit, batchno.assign_add(1)] # edited by XR: change ft to f

	
		return val_ops



def train(trains, vals,  round = 0, do_val = False, writesummary = False):
	# Train a Variational Autoencoder				batchcount = 0

	# for each round, update thetas, and P0, T0

	# Input placeholders

	Ds,  Parties, Ids = trains

	Ds_val,  Parties_val, Ids_val= vals
	print(Parties.shape)

	Ds = np.float32(Ds)
	Ds_val = np.float32(Ds_val)


	with tf.Graph().as_default():

		Dt = tf.placeholder(tf.float32, shape=[None,300])

		Ptr = tf.placeholder(tf.int64, shape = [None,])
		Idt = tf.placeholder(tf.int64, shape= [None, ])


		# train_dataset = tf.data.Dataset.from_tensor_slices((Dt,  Idt, Ptr))
		train_dataset = tf.data.Dataset.from_tensor_slices((Ds,  Ids, Parties))
		# train_dataset = train_dataset.shuffle(buffer_size = 100000)
		train_dataset = train_dataset.batch(FLAGS.batch_size)

		train_iterator = train_dataset.make_initializable_iterator()
		TrainD, TrainID, TrainParty= train_iterator.get_next()

		tf.cast(TrainD, tf.float32)

		val_dataset = tf.data.Dataset.from_tensor_slices((Ds_val,   Ids_val, Parties_val))
		val_dataset = val_dataset.batch(FLAGS.batch_size)
		val_iterator = val_dataset.make_initializable_iterator()
		ValD,  ValID, ValParty= val_iterator.get_next()
		# tf.cast(TrainD, tf.float32)
		# tf.cast(TrainA, tf.float32)


		train_ops, summary_ops = model(TrainD, TrainParty, TrainID)
		saver = tf.train.Saver()
		val_ops= model(ValD, ValParty, ValID, trained = False, reuse = True)

		init_op = tf.global_variables_initializer()
		# with tf.InteractiveSession() as sess:
		with tf.Session() as sess:
			sess.run(init_op)
			# if writesummary:
			train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train/{}/'.format(round), sess.graph)
			print('start training')
			for epoch in range(FLAGS.n_epochs):
				print("start epoch {}".format(epoch))

				# sess.run(train_iterator.initializer, feed_dict = {Dt:Ds, Ptr:Parties, Idt:Ids})
				sess.run(train_iterator.initializer)
				batchcount = 0

				while True:
					try:
						if not writesummary: # only write summary for some rounds
							# _, TrainID_, TrainParty_,T_, theta_n_, theta0_, f_, ft_, batchcount= sess.run(train_ops) # Ds, As, Thetas, Parties,Ids
							# edited by XR, remove f_ and change ft_ to f_
							_, s, p, sl, batchcount= sess.run(train_ops) # Ds, As, Thetas, Parties,Ids


						else:
							if (batchcount+1)%FLAGS.print_every ==0:
								# print(batchcount)
								# print('write summpary')

								# summary_str, elbo_, mse_, kl_f_,  _, TrainID_, TrainParty_,T_, theta_n_, theta0_, f_, ft_, batchcount = sess.run(summary_ops)
								summary_str, _, batchcount = sess.run(summary_ops) # edited by XR: change f_ to z_ and ft_ to f_



								######################
								train_writer.add_summary(summary_str, batchcount)

							else:
								# a = sess.run(train_ops)

								_, s, p, sl,batchcount = sess.run(train_ops)  # Ds, As, Thetas, Parties,Ids
								# print(s)
								# print(p)
								# print(sl)
								# break



					except tf.errors.OutOfRangeError:

						break
			
			if do_val:
				print('beging evaluation')
				# val_writer = tf.summary.FileWriter(FLAGS.logdir + '/val/{}/'.format(round), sess.graph)
				sess.run(val_iterator.initializer)
				# topics_val = []
				val_ids = []
				val_Zmus = []

				val_parties = []

				val_slants = []
				sloss  = []

				while True:
					try:
						Sloss_, ValID_, ValParty_, z_, S_pred_, batchcount = sess.run(val_ops) # edited by XR: change ft_ to f_, remove S_pred_
						sloss.append(Sloss_)
						val_Zmus.extend(z_)
						val_ids.extend(ValID_)
						val_parties.extend(ValParty_)

						val_slants.extend(S_pred_)
						# val_Z_pmus.extend(Zp_mus)


					except tf.errors.OutOfRangeError:
						break
				val_ids = np.array(val_ids)
				val_parties = np.array(val_parties)
				val_slants = np.array(val_slants)

				val_Zmus = np.array(val_Zmus)
				print('average slant loss in val', np.mean(sloss))


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

		# all zp_mus in val will be the sames 
		return  val_Zmus,  val_ids, val_parties, val_slants #edited by XR: save Frames later
	else:
		# speechids = np.array(speechids)
		# Zmus = np.array(Zmus)
		return None



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



	#########################################################################

	dws = np.load(datapath + 'dws_2009_20.npy')

	congress_df = pd.read_csv(r'D:\projects\congress\processed\processed_reports\crs_2009_20_10_04_trancated.csv', sep ='\t', encoding = 'ISO-8859-1', index_col = 0)
	As_all = np.load(datapath+ 'author_dummy_2009_20.npy')
	congress_df['author'] = list(As_all)
	congress_df['dw'] = dws
	
	congress_flt_ = congress_df.loc[~congress_df['true_id'].isin(true_switchers)]
	congress_flt_['is_Dem'] = (congress_flt_['party'] == 'D')*1
	congress_flt = congress_flt_.sort_values(by = ['dw'], ascending = True, ignore_index = True)
	ds_ns = np.load(r'D:\projects\congress\processed\training_vectors\all_berts_vec_09_20_noswitcher_sortedbydw.npy')
	ids_ns = congress_flt['speech_id'].values
	parties_ns = congress_flt['is_Dem'].values
	thetas_ns = congress_flt['topics_prior'].values
	As_ns = np.array(congress_flt['author'].values.tolist())

	# if not FLAGS.biased:
	# 	sss = StratifiedShuffleSplit(n_splits=5, test_size= 1- notmask)
	# 	train_id, test_id= next(sss.split(thetas_ns, parties_ns))
	# else:
	# 	print('start biased sampling')
	# 	orderid = np.arange(len(parties_ns))
	# 	tbn = int(len(parties_ns) * notmask/2)
	# 	train_id = np.concatenate([orderid[:tbn], orderid[-tbn:]])
	# 	test_id = orderid[tbn:-tbn]
	# 	np.random.shuffle(train_id)
	# 	np.random.shuffle(test_id)
	# print(len(train_id), len(test_id))


	# ids, ids_val = ids_ns[train_id], ids_ns[test_id]
	# ds, ds_val = ds_ns[train_id], ds_ns[test_id]
	# # As, As_val = As_ns[train_id], As_ns[test_id]
	# # thetas0, thetas_val = thetas_ns[train_id], thetas_ns[test_id]
	# parties, parties_val = parties_ns[train_id], parties_ns[test_id]
	# As, As_val = As_ns[train_id], As_ns[test_id]








	####################################################################################################



	

	print('loaded data')
	# t_accs = []
	mutualinfos=[]
	s_accs = []
	p_accs = []
	# Party_test_id = np.matmul(As_val.T, parties_val)>0
	#### on the even round update T, with orthogonal constrainst over P
	### on the odd round, fix T, slightly push F to orthogonal
	for round in range(FLAGS.alter_rounds):
		print('round {}'.format(round))


		do_val = True

		writesummary = False
		if not FLAGS.biased:
			sss = StratifiedShuffleSplit(n_splits=5, test_size= 1- FLAGS.notmask)
			train_id, test_id= next(sss.split(thetas_ns, parties_ns))
		else:
			# print('start biased sampling')
			orderid = np.arange(len(parties_ns))
			tbn = int(len(parties_ns) * FLAGS.notmask/2)
			train_id = np.concatenate([orderid[:tbn], orderid[-tbn:]])
			test_id = orderid[tbn:-tbn]
			np.random.shuffle(train_id)
			np.random.shuffle(test_id)
		print(len(train_id), len(test_id))


		ids, ids_val = ids_ns[train_id], ids_ns[test_id]
		ds, ds_val = ds_ns[train_id], ds_ns[test_id]
		# As, As_val = As_ns[train_id], As_ns[test_id]
		# thetas0, thetas_val = thetas_ns[train_id], thetas_ns[test_id]
		parties, parties_val = parties_ns[train_id], parties_ns[test_id]
		As, As_val = As_ns[train_id], As_ns[test_id]




		trains = [ds,  parties, ids]
		vals = [ds_val, parties_val, ids_val]
		Party_test_id = np.matmul(As_val.T, parties_val)>0
		if do_val:
			val_Zmus, val_ids, val_parties, val_slants = train(trains, vals, round = round, writesummary = writesummary, do_val = do_val)

			print('check doc parties prediction')
			s_acc =sum ((val_slants>0) ==parties_val)/len(parties_val)
			print('slant accuracy is {} at round {} in test'.format(s_acc, round))

			s_accs.append(s_acc)


			print("check membership prediction")

			p_acc = predict_party_membership(val_slants, Party_test_id, As_val)
			print('party membership accuracy is {} at round {} in test'.format(p_acc, round))
			p_accs.append(p_acc)


		Parties = np.array(val_parties)
		# Frames = np.array(val_Frames)
		Zmus = np.array(val_Zmus) # collection of document z_mu

		Party_by_id= np.matmul(As_val.T, Parties)>0 # if 0 then from R or no speeches, will differenitiate



		Zbar = np.matmul(As_val.T, Zmus)/(np.sum(As_val, axis = 0)[:, np.newaxis]+0.000001)


		mutualinfo = PCA_mutual_info(Zbar, Party_by_id, As_val)
		# mutualinfo  = PCA_mutual_info(Zbar, Senate_by_id, As_val)
		print("Mutual information between Zbar clusters and parties D1/R0 is :", mutualinfo)
		# print('mutual info between Zbar and senators', mutualinfo)

		mutualinfos.append(mutualinfo)

		# np.save(FLAGS.resultdir + '/speechids_{}'.format(round), val_ids) # to check if orders are wrong
		# np.save(FLAGS.resultdir + '/Zmus_{}'.format(round), Zmus)
		# np.save(FLAGS.resultdir + '/Slants_pred_{}'.format(round), val_slants)
		# np.save(FLAGS.resultdir + '/Zbars_{}'.format(round), Zbar)
	# np.save(FLAGS.resultdir + '/Frames_f_{}'.format(round), Frames)
	# np.save(FLAGS.resultdir + '/mutual_info_scores', np.array(mutualinfos))

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
		parameterdict = {"masks":[ 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03, 0.01],  'epochs':[10]} # epochs >=5
		parameterdict = {"masks":[  0.05, 0.03, 0.01],  'epochs':[10]} # epochs >=5

		parametergrid = [[i, j] for i in parameterdict['masks'] for j in parameterdict['epochs']]
		# parametergrid = [[10, 2, 0.08], [10, 3, 0.1]]
		print(parametergrid)
		all_MIs= []
		# f = open(FLAGS.resultdir+'all_mutual_info_{}.txt'.format(vs), 'w') 
		oldmodelp = FLAGS.modelpath
		oldresultp = FLAGS.resultdir
		oldlogp = FLAGS.logdir
		# print(parametergrid)
		with open(FLAGS.resultdir+'all_mutual_info_{}.txt'.format(vs), 'w') as fo:
			fo.write('train baseline simple MLP {}, biased sample is {}\n'.format(vs, FLAGS.biased))
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
