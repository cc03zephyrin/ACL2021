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
import imblearn
from imblearn.over_sampling import SMOTE


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
vs = 100.03# vamprior, adding masked labels, randomly masked
penalty = 1
rp = 5


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


def model(TrainD, TrainID, TrainParty,trained = True, reuse = False):
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

		# Merge all the summaries

		# train_ops = [train_op, TrainID, TrainParty, T, theta_n, TrainThe, f, ft,batchno.assign_add(1)]
		train_ops = [train_op,  batchno.assign_add(1)] # edited by XR: delete f, change ft to f


		return train_ops
	else:
	
		SlantLoss = tf.losses.sigmoid_cross_entropy(multi_class_labels=TrainParty, logits=s_logit)# 
	
		val_ops = [TrainID, s_logit, batchno.assign_add(1)] # edited by XR: change ft to f

	
		return val_ops





def train(trains, tests,  round = 0, do_val = False, writesummary = False):
	# Train a Variational Autoencoder				batchcount = 0

	# for each round, update thetas, and P0, T0

	# Input placeholders

	Ds,  Parties, Ids = trains

	Ds_test, Parties_test, Ids_test = tests
	print(Parties.shape)

	Ds = np.float32(Ds)
	Ds_test = np.float32(Ds_test)

	C = FLAGS.num_comp
	round = FLAGS.round
	model_dir = FLAGS.modelpath
	print(model_dir)
	model_name = '/msk_0.05_epc_10/{}/vae_{}'.format(round, round)
	model_path = model_dir + model_name
	print(model_path)
	tf.reset_default_graph()


	with tf.Graph().as_default():


		Dt = tf.placeholder(tf.float32, shape=[None,300])

		Ptr = tf.placeholder(tf.int64, shape = [None,])
		Idt = tf.placeholder(tf.int64, shape= [None, ])


		train_dataset = tf.data.Dataset.from_tensor_slices((Dt,  Idt, Ptr))
		# train_dataset = train_dataset.shuffle(buffer_size = 100000)
		train_dataset = train_dataset.batch(FLAGS.batch_size)

		train_iterator = train_dataset.make_initializable_iterator()
		TrainD, TrainID, TrainParty= train_iterator.get_next()

		tf.cast(TrainD, tf.float32)

		test_dataset = tf.data.Dataset.from_tensor_slices((Ds_test, Ids_test, Parties_test))
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		TstD, TstID, TstParty = test_iterator.get_next()
		# tf.cast(TrainD, tf.float32)
		# tf.cast(TrainA, tf.float32)
		

		train_ops= model(TrainD,  TrainID, TrainParty)
		saver = tf.train.Saver()
		test_ops = model(TstD,  TstID, TstParty,  trained = False, reuse = True)

		# all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
		# print(all_variables)
		
		init_op = tf.global_variables_initializer()
		
		# with tf.InteractiveSession() as sess:
		with tf.Session() as sess:
			sess.run(init_op)
			saver.restore(sess = sess, save_path = model_path)

			print('start training')
			for epoch in range(FLAGS.n_epochs):
				print("start epoch {}".format(epoch))

				sess.run(train_iterator.initializer, feed_dict = {Dt:Ds,  Ptr:Parties, Idt:Ids})
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
			# tst_thetas = []
			# tst_parties = []
			tst_slants = []
			while True:
				try:
					TstID_,   S_pred_,batchcount = sess.run(test_ops) 
					tst_ids.extend(TstID_)
					tst_slants.extend(S_pred_)
					# tst_Zmus.extend(z_)
					# tst_thetas.extend(theta_t_)
					# tst_parties.extend(TstParty_)
				except tf.errors.OutOfRangeError:
					break
			tst_ids = np.array(tst_ids)
			tst_slants = np.array(tst_slants)
			print(tst_slants)
			# print(tst_ids)
			# print(Ids_test)

			# tst_thetas = np.array(tst_thetas)
			assert (tst_ids==Ids_test).all(), 'test ids no match, check id order to make sure order is not changed'

	

	if do_val:
		


		# Zmus = np.array(Zmus)
		# all zp_mus in val will be the sames 
		return  tst_slants #edited by XR: save Frames later



def iterative_training():
	# load data, initialize P0 and T0

	datapath = FLAGS.data_dir + 'debabe_test/'
	ds_all = np.load(os.path.join(datapath, 'debate_sentencevector.npy'))

	enddate = '2020-10-04'
	# ds_all = np.load(np.load(os.path.join(datapath, 'cong_reports_102_114_for_training_sif.npz'))['arr_0'])


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

	else:
		dws_all = np.float32(np.load(os.path.join(datapath, 'cr_authordw_bydoc_1991_{}.npy'.format(enddate))))[inx]

		orderid = np.argsort(dws_all[:, 0])
		tbn = int(len(parties_all) * FLAGS.notmask/2)
		train_id = np.concatenate([orderid[:tbn], orderid[-tbn:]])
		test_id = [i for i in range(len(orderid)) if i not in train_id]
		mask_[train_id] = 1
		mask_pre = mask_.copy()



	ids, ids_test = ids_all[train_id], ids_all[test_id]
	ds, ds_test = ds_all[train_id], ds_all[test_id]
	# As, As_test = As_all[train_id], As_all[test_id]
	# thetas0, thetas_test = thetas_all[train_id], thetas_all[test_id]
	parties, parties_test = parties_all[train_id], parties_all[test_id]
	print(len(train_id), len(test_id))
	print(parties.mean(), parties_test.mean())
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
	# t_accs = []
	# mutualinfos=[]
	s_accs = []


	for round in range(FLAGS.alter_rounds):
		print('round {}'.format(round))

	
		do_val = True


		print('shuffle data')
		index = np.arange(len(ds))
		np.random.shuffle(index)
		ds = ds[index]

		parties = parties[index]
		ids = ids[index]
		print('shape after shuffle', ds.shape, len(index))
		# sort the input, non masked first



		trains = [ds,  parties, ids]
	
		tests = [ds_test,  parties_test, ids_test]
		if do_val:
			tst_slants = train(trains, tests,  round = round,  writesummary = False, do_val = do_val)


			print('check doc parties prediction')
			s_acc =sum ((tst_slants>0) ==parties_test)/len(parties_test)
			print(parties_test.mean())
			print('slant accuracy is {} at round {} in test'.format(s_acc, round))
			print((tst_slants>0))
			print((tst_slants>0).sum())
			# t_accs.append(t_acc)
			s_accs.append(s_acc)




	return  s_accs


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
		parameterdict = {"masks":[0.8, 0.6,0.4,0.2, 0.1, 0.08,0.05, 0.03, 0.01],  'epochs':[20]} # epochs >=5
		# parameterdict = {"masks":[0.6],  'epochs':[10]} # epochs >=5

		# parametergrid = [[i, j , k] for i in parameterdict["kl_anneal"] for j in parameterdict['beta_z'] for k in parameterdict['epochs'] if (k <i) and (k>=i/2)]
		# parametergrid = [[10, 2, 0.08], [10, 3, 0.1]]
		parametergrid = [[i, j] for i in parameterdict['masks'] for j in parameterdict['epochs']]
		print(parametergrid)

		# oldmodelp = FLAGS.modelpath
		oldresultp = FLAGS.resultdir
		# oldlogp = FLAGS.logdir
		# print(parametergrid)

		for msk, epc in parametergrid:
			print(msk, epc)

			# FLAGS.anneal_kl = float(ann)
			FLAGS.notmask = msk
			# FLAGS.num_comp = int(com)
			FLAGS.n_epochs = epc
			versname = "msk_{}_epc_{}".format(msk, epc)



			s_accs = iterative_training()
	

			# fig = plt.figure()
			# ax1 = fig.add_subplot(111)
			# a = len(s_accs)
			# # ax1.plot(np.arange(a), t_accs, '.r-',  label='topics')
			# # ax1.plot(np.arange(a), t_accs, )
			# # ax1.plot(np.arange(a), MIs, 'xb-', label='MutualInfos')
			# ax1.plot(np.arange(a), s_accs, 'g-', label='doc slants')
			# plt.legend(loc='upper left');
			# ax1.set_title('training evaluation version {}'.format(vs))
			# plt.savefig(FLAGS.resultdir + 'full sample evaluation_{}.jpg'.format(versname))
			# plt.close()
			f = open(oldresultp+'all_acc_{}.txt'.format(vs), 'a') 

			f.write(versname)
			f.write('\t')

			f.write(str(s_accs[0]))
			# f.write('\t')
			# f.write(str(list(t_accs)))
			f.write('\n')
			print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
			print(versname, s_accs)
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
