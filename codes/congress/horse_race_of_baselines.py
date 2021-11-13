
# coding: utf-8

# In[60]:


# this noteboke is to run several horse races, it will contain a sampling function taking input of texts and outputs of labels and authors
# performing biased and unbiased sampling


import itertools
import matplotlib as mpl
import numpy as np
import os
import re

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
import scipy

# from tensorflow.contrib.layers import fully_connected
from matplotlib import pyplot as plt
import tensorflow_probability as tfp
import math
from sklearn.decomposition import PCA
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords

import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer



# In[61]:



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Concatenate, GRU
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import re


# In[3]:


# define global variables

datapath = r'/home/cczephyrin/projects/political embedding/training_vec/'


# In[3]:


# load data
# df = pd.read_csv(r'/home/cczephyrin/projects/political embedding/data/crs_1991_2020-10-04_sorted_filtered.csv', sep = '\t', encoding= 'ISO-8859-1', index_col =0)


# In[7]:


def process_text(x, CUSTOM_FILTERS = [strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords]):

    regex = re.compile('[^a-zA-Z ]')
    corpus = regex.sub('', x)
#     print(regex.sub("", 'era 23'))
    corpus = re.sub('\s+', ' ', corpus)

    sentences = preprocess_string(corpus, CUSTOM_FILTERS)
    return " ".join(sentences)
    


# In[23]:


# df['processed_speech'] = df['speech'].apply(lambda x: process_text(x))



# In[25]:


# df.head()


# In[4]:


true_switchers_ = ['A000361','B000229', 'B001264', 'D000168', 'G000280', 'H000067', 'H000390', 'L000119', 'P000066', 'T000058', 'C000077','F000257', 'G000557', 
'S000320', 'S000709','J000072']

independents = ['S000033', 'B001237', 'K000383']

true_switchers =independents + true_switchers_
enddate = '2020-10-04'
# bottom= FLAGS.bottom

date = np.load(os.path.join(datapath, 'sorted_date_1991_{}.npy'.format(enddate)))

inx = date>20090000


#########################################################################

print(date.shape)



# In[65]:


# filter data and save them
# data_baseline = df.loc[inx, ['processed_speech', 'speech_id']]
# 


# In[5]:


# data_baseline.to_csv(datapath + 'prcoessed_text_baseline.csv', sep ='\t', encoding = 'ISO-8859-1', index_label = False)
data_baseline = pd.read_csv(datapath + 'prcoessed_text_baseline.csv', sep ='\t', encoding = 'ISO-8859-1', index_col = 0)


# In[6]:


# define sampling functions, load As, dws, allow biased unbiased sampling

As_all = np.load(os.path.join(datapath, 'cr_author_dummy_1991_{}.npy'.format(enddate)))[inx]

parties_all = np.load(os.path.join(datapath, 'cr_parties_1991_{}.npy'.format(enddate)))[inx]

As_ids = np.load(os.path.join(datapath, 'cr_author_id_1991_{}.npy'.format(enddate)),allow_pickle=True)

dws_all = np.float32(np.load(os.path.join(datapath, 'cr_authordw_bydoc_1991_{}.npy'.format(enddate))))[inx][:, 0]


# In[8]:


# data_baseline.shape, dws_all.shape, As_all.shape


# In[114]:


def sample(biased = True, notmask = 0.8):


    switchdummies = [list(As_ids).index(i) for i in true_switchers if i in As_ids]

    switch_speech_bool = As_all[:, switchdummies].sum(axis = -1)==1

    # sum(switch_speech_bool),len(switch_speech_bool)

    switch_id = np.arange(len(switch_speech_bool))[switch_speech_bool]
    train_id_wo_switchers = np.setdiff1d(np.arange(len(As_all)), switch_id) # for model eval, exclude switchers, add them back for insights
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
    # train_id, test_id = next(sss.split(As_all, parties_all))


    # train_id_wo_switchers = np.setdiff1d(train_id, switch_id)
    #test_id_wo_switchers = np.union1d(test_id, switch_id)

    # test_id_wo_switchers = np.setdiff1d(test_id, switch_id)

#     print(sum(switch_speech_bool),len(switch_speech_bool))

    As_ns = As_all[train_id_wo_switchers]

    parties_ns = parties_all[train_id_wo_switchers]
    dws_ns = dws_all[train_id_wo_switchers]


    if not biased:
        sss = StratifiedShuffleSplit(n_splits=5, test_size= 1- notmask)
        train_, test_= next(sss.split(As_ns, parties_ns))
        train_id, test_id = train_id_wo_switchers[train_], train_id_wo_switchers[test_]
        print(len(train_id), len(test_id))
    else:
        print('start biased sampling')
        orderid = np.argsort(dws_ns)
        train_id_wo_switchers_or = train_id_wo_switchers[orderid]
        tbn = int(len(parties_ns) * notmask/2)
        train_id = np.concatenate([train_id_wo_switchers_or[:tbn], train_id_wo_switchers_or[-tbn:]])
        test_id = train_id_wo_switchers_or[tbn:-tbn]
        print(len(train_id), len(test_id))
        train_id = shuffle(train_id)
        test_id = shuffle(test_id)
    return train_id, test_id


# In[115]:


# train_id, test_id = sample(biased = False)


# In[111]:


# train_id, test_id


# In[112]:


# start training on models

data= data_baseline['processed_speech'].values
pty = parties_all

del data_baseline
# In[113]:


# pty[test_id].mean()


# In[94]:


# for n in np.arange(0,100, 5):
#     print(n)
#     print(np.percentile(xlens, n))


# In[90]:


# pty


# In[116]:


# bi-directional lstm

def bi_lstm_sample(data, pty, maxlen = 300, max_features = 100000):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data)
    X = tokenizer.texts_to_sequences(data)
    X = pad_sequences(X, maxlen= maxlen)
    y = pty
    return X, y


# In[133]:


def bi_lstm(X, y, train_id, test_id,  batch_size = 500, n_epoch = 10, max_features = 100000, trial = False): # 80% padding, 20% trancate
    if trial:
        num = -1000
        nu = -100
        n_epoch = 1
    else:
        num = 0
        nu= 0
    X_train, X_test = X[train_id][num:], X[test_id][nu:]
    maxlen = X.shape[1]
    

    print("train, test shape", X_train.shape, X_test.shape)
    y_train, y_test = y[train_id][num:], y[test_id][nu:]
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(max_features, 300, input_length=maxlen)(sequence)
    forwards = GRU(100)(embedded)
#     backwards = LSTM(100, go_backwards=True)(embedded)
#     merged = Concatenate(axis = -1)([forwards, backwards])
    after_dp = Dropout(0.5)(forwards)
    inter = Dense(100, activation = 'relu')(after_dp)
    output = Dense(1, activation='sigmoid')(inter)
    model = Model(sequence, output)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    print('Train...')
    model.fit(X_train, y_train, 
          batch_size=batch_size,
          epochs=n_epoch)
    

    
    y_= model.predict(X_test)
    y_pred= np.squeeze(y_)>=0.5
    # print(y_)
    # print(y_test)
    acc = (y_pred == y_test).mean()
    print('test accuracy', acc)

    # party accu
    valid = As_all[test_id][nu:].T.sum(axis = 1)>0
    party_p = np.matmul(As_all[test_id][nu:].T[valid], y_pred)/np.sum(As_all[test_id][nu:].T[valid], axis = 1)>0
    party_t = np.matmul(As_all[test_id][nu:].T[valid], y_test)/np.sum(As_all[test_id][nu:].T[valid], axis = 1)
    assert np.array_equal(party_t, party_t.astype(bool)), 'party error'

    pcc = np.mean(party_p==party_t)
    return acc, pcc

## def RF function

def RF_sample(data, pty, max_features = 5000):
    vector = CountVectorizer(max_features = max_features, dtype = np.uint16)
    vector.fit(data)
    a = vector.transform(data)
    return a.toarray(), pty

def RF_train(X, y, train_id, test_id, trial = False, max_features = 5000):
    if trial:
        num = -1000
        nu = -100
        n_epoch = 1
    else:
        num = 0
        nu= 0
    X_train, X_test = X[train_id][num:], X[test_id][nu:]

    print("train, test shape", X_train.shape, X_test.shape)
    y_train, y_test = y[train_id][num:], y[test_id][nu:]
    model = RandomForestClassifier(n_estimators=100, max_depth = 10, min_samples_split = 50,  min_samples_leaf = 10, 
                               warm_start=True, oob_score=True)
    model.fit(X_train, y_train)
    accuracy = model.oob_score_
    print('oob acc is ', accuracy)
    y_pred = model.predict(X_test)
    # print(y_pred)
    acc = (y_pred==y_test).mean()

    # party accu
    valid = As_all[test_id][nu:].T.sum(axis = 1)>0
    party_p = np.matmul(As_all[test_id][nu:].T[valid], y_pred)/np.sum(As_all[test_id][nu:].T[valid], axis = 1)>0
    party_t = np.matmul(As_all[test_id][nu:].T[valid], y_test)/np.sum(As_all[test_id][nu:].T[valid], axis = 1)
    assert np.array_equal(party_t, party_t.astype(bool)), 'party error'

    pcc = np.mean(party_p==party_t)
    return acc, pcc

def repeated_test(data, pty, repeat= 5, method = 'gru', model_funcs = [bi_lstm_sample, bi_lstm], biased = True, notmask = 0.05):
    accs = []
    pccs = []
    global max_features
    global toy
    dataf, trainf = model_funcs
    print('begin training {}'.format(method))
    X, y = dataf(data, pty, max_features = max_features)
    accs = []

    for i in range(repeat):
        train_id, test_id = sample(biased = biased, notmask = notmask)

        acc, pcc = trainf(X, y, train_id, test_id, trial = toy, max_features = max_features)
        accs.append(acc)
        pccs.append(pcc)
    return accs, pccs
        


# In[157]:


def experiments(data, pty,models_names = ['gru'], models_dict = {'gru': [bi_lstm_sample, bi_lstm]}, notmasks = [0.01, 0.03, 0.05, 0.08, 0.2, 0.4, 0.6, 0.8]):
    for md in models_names:
        for biased in [True, False]:
            bia = 'bias'
            if not biased:
                bia = 'unbias'
            f = open( r'/home/cczephyrin/projects/political embedding/results/baselines/{}_{}.csv'.format(md, biased), 'w')


            for notmask in notmasks:
                
                version = "{}_{}_{}".format(md, notmask, bia)

                print(version)
                model_funcs = models_dict[md]
                accs, pccs = repeated_test(data, pty, method = md, model_funcs = model_funcs, biased = biased, notmask= notmask)
                print(accs)
                f.write(version)
                f.write('\t'+"pcc is")
                for pc in pccs:
                    f.write('\t'+ str(pc))
                f.write('\t'+"acc is")
                for ac in accs:
                    f.write('\t'+ str(ac))
                f.write('\n')
            f.close()



# In[158]:


models_names = ['rf']
# models_names = ['rf', 'gru']
# models_names = [ 'gru']
models_dict = {'gru': [bi_lstm_sample, bi_lstm], 'rf': [RF_sample, RF_train]}

# set toy = False
toy= False
# toy = True
# max_features = 50000
max_features = 30000

notmasks = [0.01, 0.03, 0.05, 0.08, 0.2, 0.4, 0.6, 0.8]
# notmasks = [ 0.6]
experiments(data, pty,  models_names = models_names, models_dict = models_dict, notmasks = notmasks)

