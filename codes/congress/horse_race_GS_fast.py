
# coding: utf-8

# In[151]:


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
from sklearn.linear_model import Lasso, Ridge
import json
import re
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from nltk.util import bigrams
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle
import en_core_web_sm



# In[2]:


# define global variables

datapath = r'/home/cczephyrin/projects/political embedding/training_vec/'


# In[3]:


# load data
# df = pd.read_csv(r'/home/cczephyrin/projects/political embedding/data/crs_1991_2020-10-04_sorted_filtered.csv', sep = '\t', encoding= 'ISO-8859-1', index_col =0)


# In[4]:


def process_text(x, CUSTOM_FILTERS = [strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords]):

    regex = re.compile('[^a-zA-Z ]')
    corpus = regex.sub('', x)
#     print(regex.sub("", 'era 23'))
    corpus = re.sub(r'\s+', ' ', corpus)

    sentences = preprocess_string(corpus, CUSTOM_FILTERS)
    return " ".join(sentences)
    


# In[5]:


# df['processed_speech'] = df['speech'].apply(lambda x: process_text(x))



# In[6]:


# df.head()


# In[160]:


true_switchers_ = ['A000361','B000229', 'B001264', 'D000168', 'G000280', 'H000067', 'H000390', 'L000119', 'P000066', 'T000058', 'C000077','F000257', 'G000557', 
'S000320', 'S000709','J000072']

independents = ['S000033', 'B001237', 'K000383']

true_switchers =independents + true_switchers_
enddate = '2020-10-04'
# bottom= FLAGS.bottom

date = np.load(os.path.join(datapath, 'sorted_date_1991_{}.npy'.format(enddate)))

# inx = date>20200600
inx = date> 20090000
to_remove = [('do', 'nt'), ('mr', 'speaker'), ('house', 'representative'), ('year', 'ago'), ('honor', 'life'),('speaker', 'rise'),
             ('speaker', 'pro'),('pro', 'tempore'),('ask', 'colleague'), ('th', 'anniversary'), ('give', 'permission'),
            ('ms', 'speaker'),('colleague', 'join'),('honor', 'recognize'),('extend', 'remark'),('today', 'recognize'),
            ('yea', 'roll'),('roll', 'yea'),('speaker', 'today'),('roll', 'nay'),('nay', 'roll'),('wo', 'nt'),
            ('want', 'thank'), ('permission', 'address'), ('rise', 'today'),('would', 'nt'), ('should', 'nt'), ('nominations', 'beginning'),  ('objection', 'ordered'), ('ask', 'unanimous'), 
            ('unanimous', 'consent'),  ('authorized', 'meet'),  ('meet', 'session'),('consent', 'senate'), ('senate', 'proceed'),
             ('conduct', 'hearing'), ('legislative', 'clerk'),('congressional', 'record'), ('senate', 'appeared'), ('appeared', 'congressional'), ('received', 'senate'), ('nominations', 'received'), ('session', 'senate')]
# inx = [1000:]


#########################################################################





# In[161]:


# filter data and save them
# data_baseline = df.loc[inx, ['processed_speech', 'speech_id']]
# 


# In[162]:


# data_baseline.to_csv(datapath + 'prcoessed_text_baseline.csv', sep ='\t', encoding = 'ISO-8859-1', index_label = False)
data_baseline = pd.read_csv(datapath + 'prcoessed_text_baseline.csv', sep ='\t', encoding = 'ISO-8859-1', index_col = 0)


# In[164]:


# data_baseline = data_baseline.iloc[-2637:,:]   
# data_baseline


# In[165]:


# define sampling functions, load As, dws, allow biased unbiased sampling

As_all = np.load(os.path.join(datapath, 'cr_author_dummy_1991_{}.npy'.format(enddate)))[inx]

parties_all = np.load(os.path.join(datapath, 'cr_parties_1991_{}.npy'.format(enddate)))[inx]

As_ids = np.load(os.path.join(datapath, 'cr_author_id_1991_{}.npy'.format(enddate)),allow_pickle=True)

dws_all = np.float32(np.load(os.path.join(datapath, 'cr_authordw_bydoc_1991_{}.npy'.format(enddate))))[inx][:, 0]


# In[166]:


As_num= np.argmax(As_all, axis = 1)
As_num.shape, As_num


# In[167]:


data_baseline['last_name'] = As_num
data_baseline['last_name']


# In[168]:


data_baseline['isdemocrat'] = parties_all
data_baseline = data_baseline.reset_index(drop= True)
data_baseline.rename(columns = {'processed_speech': 'text'}, inplace = True)


# In[169]:


nlp = spacy.load('en_core_web_sm')

def tokenizer(x):
    return [
        item.lemma_.lower() for item in nlp(x, disable=['parser',  'ner', 'senter', 'tok2vec']) ]
# In[170]:


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


# In[171]:

# 
# train_id, test_id = sample(biased = True)


# In[131]:


def get_usefulbigram(test_df,minfreq= 20, topn = 10000):
    t_text = test_df['text'].values
    t_party = test_df['isdemocrat'].values
    len_t = []
#     len_r = 0 
#     len_d = 0 
    tcounter = Counter()
    dcounter = Counter()
    rcounter = Counter()
    for p, sentence in tqdm(enumerate(t_text)):
        tokens = tokenizer(sentence)
        pty = t_party[p]
        all_bigrams = list(bigrams(tokens))
        tcounter.update(all_bigrams)
        len_t.append(len(all_bigrams))
        if pty:
            dcounter.update(all_bigrams)
#             len_d += len(all_bigrams)
        else:
            rcounter.update(all_bigrams)
#             len_r+= len(all_bigrams)
        
#     print(tcounter.keys())

    can_bigram = [b for b in tcounter.keys() if tcounter[b] >= minfreq if b not in to_remove]

    d_freq = np.array([dcounter[bg] for bg in can_bigram])
    len_d = d_freq.sum()
    d_rest = len_d -d_freq
    r_freq = np.array([rcounter[bg] for bg in can_bigram])
    len_r = r_freq.sum()
    r_rest = len_r - r_freq
    kisq = (d_freq*d_rest - r_freq * r_rest)**2 / ((d_freq+r_freq)*(d_freq+d_rest)*(d_rest+r_rest)*(r_freq+ r_rest))
    kiind = np.argsort(-kisq)[:topn]
    useful_bigram = [can_bigram[i] for i in kiind]
    return useful_bigram

# In[132]:



# useful_bigram = get_usefulbigram(test_df)


# In[153]:


# def sentence_to_bigram(text):
# #     try:
# #     global useful_bigram
# #     except:
# #     nonlocal useful_bigram
#     tokens = tokenizer(text)
#     bg =[bg for bg in bigrams(tokens) if bg in useful_bigram]
#     return bg


# In[154]:


# data_baseline['bigrams'] = data_baseline['text'].apply(lambda x: sentence_to_bigram(x))


# In[155]:


# data_baseline


# In[172]:


def gs_model(data_baseline, train_id, test_id, useful_bigram):
    train_df = data_baseline.loc[train_id,['last_name', 'text', 'isdemocrat', 'bigrams']]
    test_df = data_baseline.loc[test_id,['last_name', 'text', 'isdemocrat', 'bigrams']]
    bigram_stat={}
    for bigram in useful_bigram:
        bigram_stat[bigram] = []
    groups = train_df.groupby('last_name')
    print('start get bigram in train.....')
    for key, gp in groups:
        #   print(key)
        bigram_per_pol= 0
        #   bigram_top = {}
        trp_counter = Counter()
        for index, entry in gp.iterrows():
            all_bigrams = entry['bigrams']
            bigram_per_pol += len(all_bigrams) # total number of bigram per person
            trp_counter.update(all_bigrams)
        yp =  2 * gp['isdemocrat'].values[0]-1
        for k in useful_bigram:
            if k not in trp_counter.keys():
                fr = 0

            else:
                fr = (1.0*trp_counter[k])/(bigram_per_pol*1.0)

            bigram_stat[k].append([yp, fr])
    
    # get coeef
    print('get coef')
    # print(bigram_stat['african', 'american'])
    # all_a_t = []
    # all_b_t = []

    # for bigram in useful_bigram:
    #     x = []
    #     y = []
    # #     w = []


    #     for entry in bigram_stat[bigram]:
    #         if entry:
    #             y.append(entry[-1])

    #             x.append(entry[:-1])





    #     clf = Ridge(alpha=0.01)
    # #     if not x:
    # #       print('ha')
    # #       print(bigram)
    # #       all_a_t.append([0])
    # #       all_b_t.append(0)
    # #       continue
    #     clf.fit(x, y)
    #     all_a_t.append(clf.coef_)
    #     all_b_t.append(clf.intercept_)
    # print(len(all_b_t))
    
    
    bigram_coef = {}

    for bigram in useful_bigram:
        x = []
        y = []
    #     w = []


        for entry in bigram_stat[bigram]:
            if entry:
                y.append(entry[-1])


                x.append([entry[-2]])

    #     print(x, y)

        lr = LinearRegression()
    #     if not x:
    # #       print('ha')
    # #       print(bigram)
    #       all_a.append([0])
    #       all_b.append(0)

    #       continue
        lr.fit(x, y)
    #     print(lr.coef_[0])
    #     print(lr.intercept_)
        bigram_coef[bigram] = [lr.coef_, lr.intercept_]


    # print(all_a[0])
    # print(bigram_coef[('african', 'american')])
    
    

    print("start test...")
    td_bigams =test_df.bigrams.values
    td_label = test_df.isdemocrat.values
    # td_tp = test_df[topic_list].values
    # td_predict = [] # ridge
    td_predict_ntp = [] # regular
    y_true = []
    a = []
    b = []
    for bigram in useful_bigram:
        a.append(bigram_coef[bigram][0][0])
        b.append(bigram_coef[bigram][1])
    # print('************************check a, b fs*******************************')   
    a = np.array(a)
    
    
    b = np.array(b)


    # print(a.mean())
    # print(b.mean())
    for i, sentbigram in enumerate(td_bigams):

        l = len(sentbigram)
        sent_counter = Counter(sentbigram)

        if l==0:
            continue
        y_true.append(td_label[i])


        fs = []
        for bigram in useful_bigram:
            if bigram in sent_counter.keys():
                f = (1.0*sent_counter[bigram]) / l
            else:
                f= 0

            fs.append(f)
        fs = np.array(fs)
        f0 = (fs!=0) + 0.01 # downweight non spoken phrases


        score_ntp  = np.sum(a * (fs - b)*f0)/np.sum((a**2) *f0)
        
        # if i ==10:
        #     print("check fs")
        #     print(sent_counter.most_common(20))
        #     print(fs.mean())
        #     print(score_ntp)

        # for bigram in sen_counter.keys(): # all bigram has to be in useful bigram
        #     f = (1.0*sen_counter[bigram]) / l
        #     a = bigram_coef[bigram][0]
        #     b = bigram_coef[bigram][1]

        #     if a[0]!=0:
        #         score_ntp += (f - b)/a[0] 


        td_predict_ntp.append(score_ntp)
    y_pred = np.array(td_predict_ntp)> 0 
 

   
    # print('ridge', accuracy_score(td_label, np.array(td_predict) > 0))
    acc = accuracy_score(y_true, y_pred)
    print('lr', acc)
    # acc_ridge = accuracy_score(td_label, np.array(td_predict) > 0)



    #######################################test party
    
    testgps = test_df.groupby('last_name')
    y_true = []
    pred_sc = []
    for key, gp in testgps:
        if len(testgps)==0:
            continue
        #   print(key)
        bigram_test_pol= 0
        testpol_counter = Counter()
        for index, entry in gp.iterrows():
            senbigram = entry['bigrams']
            bigram_test_pol += len(senbigram) # total number of bigram per person
            testpol_counter.update(senbigram)
        if bigram_test_pol ==0:
            continue
        y_true.append(gp['isdemocrat'].values[0]) 
        

        fs = []
        for bigram in useful_bigram:
            if bigram in testpol_counter.keys():
                f = (1.0*testpol_counter[bigram]) / bigram_test_pol
            else:
                f= 0

            fs.append(f)
        fs = np.array(fs)
        f0 = (fs!=0) + 0.01

        score_test  = np.sum(a * (fs - b)*f0)/np.sum((a**2) *f0)


        # if bigram in testpol_counter.keys():
        #     f = (1.0*testpol_counter[bigram]) / bigram_test_pol
        #     a = bigram_coef[bigram][0]
        #     b = bigram_coef[bigram][1]

        #     if a[0]!=0:
        #         score_test += (f - b)/a[0] 

        pred_sc.append(score_test)

    pred_sc = np.array(pred_sc)
    print('test stability', pred_sc.mean(), pred_sc.std())
    pcc = accuracy_score(y_true, pred_sc > 0)
    print('acc party ', pcc)

    
    return acc, pcc
    


# In[173]:


def repeated_test(data_baseline, useful_bigram,repeat= 5, biased = True, notmask = 0.05):
    accs = []
    pccs = []


    for rd in range(repeat):
        print('begin training round {}'.format(rd))
        train_id, test_id = sample(biased = biased, notmask = notmask)
        print(train_id[:10], test_id[:10])
        # test_df = data_baseline.loc[test_id,['last_name', 'text', 'isdemocrat']]

        acc, pcc = gs_model(data_baseline, train_id, test_id,useful_bigram)
        accs.append(acc)
        pccs.append(pcc)
        

    return accs, pccs
        


# # In[175]:


# accs, pccs = repeated_test(data_baseline, repeat =1)


# # In[176]:


# print(accs, pccs)


# In[177]:


def expriments(data_baseline, notmasks = [0.01, 0.03, 0.05, 0.08, 0.2, 0.4, 0.6, 0.8], repeat  = 1 , topn = 5000):
    md = "GS_NB"

    _, tid = sample(biased = False, notmask = 0.2)
    # minfreq = 50 * math.sqrt((1- notmask)/0.2)
    minfreq = 50 * math.sqrt((1- 0.2)/0.2)
    print(minfreq)
    tdf = data_baseline.loc[tid,['last_name', 'text', 'isdemocrat']]
    useful_bigram= get_usefulbigram(tdf, minfreq = minfreq, topn = topn)
    # print(useful_bigram)


    def sentence_to_bigram(text):

        tokens = tokenizer(text)
        bgs_ = bigrams(tokens)
        bgs=[bg for bg in bgs_ if bg in useful_bigram]
        return bgs

    print('start tranform all sentence to bigrams')
    np.save(datapath+'useful_bigram_{}'.format(topn), np.array(useful_bigram))
    data_baseline['bigrams'] = data_baseline['text'].apply(lambda x: sentence_to_bigram(x))
    data_baseline.to_csv(datapath + 'prcoessed_text_bigrams.csv', sep ='\t', encoding = 'ISO-8859-1', index_label = False)
    # data_baseline = pd.read_csv(datapath + 'prcoessed_text_bigrams.csv', sep ='\t', encoding = 'ISO-8859-1', index_col = 0)

    

    for biased in [True, False]:
        bia = 'bias'
        if not biased:
            bia = 'unbias'
            repeat = 5
        f = open( r'/home/cczephyrin/projects/political embedding/results/baselines/{}_{}_final.csv'.format(md, bia), 'w')


        for notmask in notmasks:

            version = "{}_{}_{}".format(md, notmask, bia)

            print(version)

            accs, pccs = repeated_test(data_baseline, useful_bigram, biased = biased, notmask= notmask, repeat = repeat)
            print(accs)
            print(pccs)
            f.write(version)
            f.write('\t'+"pcc is")
            for pc in pccs:
                f.write('\t'+ str(pc))
            f.write('\t'+"acc is")
            for ac in accs:
                f.write('\t'+ str(ac))
            f.write('\n')
        f.close()



# In[180]:




notmasks = [0.01, 0.03, 0.05, 0.08, 0.2, 0.4, 0.6, 0.8]
# notmasks = [0.8]
repeat = 1
expriments(data_baseline, notmasks = notmasks, repeat = repeat)