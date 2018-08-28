import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = tf.Session(config=config)
KTF.set_session(session)

import FileUtil
import numpy
import time
import sys
import subprocess
import os
import random
from Represent_luo import RepresentationLayer
numpy.random.seed(123)
random.seed(12345)

import Eval

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Lambda
from keras.layers import Embedding,Input
from keras.layers import LSTM, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Average
from keras import backend as K


if __name__ == '__main__':

    s = {
         'max_len_token':600, 
         'max_len_abner':600, 
         'max_len_cui':600, 
         'mini_batch':64,
         'epochs':35,
         'each_print':True,
         'model_save':True,
         'corpus':'PPIAC_BC2'#PPIAC_BC3,PPIAC_BC2
        }

    print s
    folder = '/home/BIO/luoling/PPIAC_CNN/'
    rep1 = RepresentationLayer(
            folder + 'word2vec_model/bc_ex3_token_100.word2vec_model',\
            vec_size=100, frequency=500000)

 

#chanel1    
    chanel1_input=Input(shape=(s['max_len_token'],),dtype='int32')
    chanel1_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], \
                          weights=[rep1.vec_table], input_length=s['max_len_token'])
    chanel1_vec=chanel1_emb(chanel1_input)

    chanel1_vec=Dropout(0.4)(chanel1_vec)

    chanel1_lstm1=Bidirectional(LSTM(100, implementation=2,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))(chanel1_vec)
    chanel1_pool=GlobalMaxPooling1D()(chanel1_lstm1)

    concat_fc=Dense(200,activation='relu')(chanel1_pool)
    concat_fc=Dense(100,activation='relu')(concat_fc)
    concat_fc=Dropout(0.4)(concat_fc)
    
    predict=Dense(2,activation='softmax')(concat_fc)

    model= Model(inputs=chanel1_input,outputs=predict)
    opt = RMSprop(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()

    #data set
    Corpus=s['corpus']
    if Corpus=='PPIAC_BC3':
        test1 = [line.strip() for line in open(folder + 'corpus/bc3_ppi_test.token.lab')]
        test2 = [line.strip() for line in open(folder + 'corpus/bc3_ppi_test.abner.lab')]
        test3 = [line.strip() for line in open(folder + 'corpus/bc3_ppi_test.cui.lab')]  

        train1 = [line.strip() for line in open(folder + 'corpus/bc3_dev_ppi_train.token.lab')]
        train2 = [line.strip() for line in open(folder + 'corpus/bc3_dev_ppi_train.abner.lab')]
        train3 = [line.strip() for line in open(folder + 'corpus/bc3_dev_ppi_train.cui.lab')]
        #10% to dev, remaining to train
        dev_num=int(len(train1)*0.1)
        train=list(zip(train1,train2,train3))
        bc3_dev=train[0:4000]
        random.shuffle(bc3_dev)
        new_dev=bc3_dev[0:dev_num]
        train=bc3_dev[dev_num:]+train[4000:]
        random.shuffle(train)
        dev1,dev2,dev3=zip(*new_dev)
        dev1=list(dev1)
        dev2=list(dev2)
        dev3=list(dev3)
        train1[:],train2[:],train3[:]=zip(*train)
        print 'train-len',len(train1),'dev-len:',len(dev1)

    elif Corpus=='PPIAC_BC2':
        test1 = [line.strip() for line in open(folder + 'corpus/bc2_ppi_test.token.lab')]
        test2 = [line.strip() for line in open(folder + 'corpus/bc2_ppi_test.abner.lab')]
        test3 = [line.strip() for line in open(folder + 'corpus/bc2_ppi_test.cui.lab')]  

        train1 = [line.strip() for line in open(folder + 'corpus/bc2_ppi_train.token.lab')]
        train2 = [line.strip() for line in open(folder + 'corpus/bc2_ppi_train.abner.lab')]
        train3 = [line.strip() for line in open(folder + 'corpus/bc2_ppi_train.cui.lab')]
        #10% to dev, remaining to train
        dev_num=int(len(train1)*0.1)
        train=list(zip(train1,train2,train3))
        random.shuffle(train)
        bc2_dev=train[0:dev_num]
        train=train[dev_num:]
        dev1,dev2,dev3=zip(*bc2_dev)
        dev1=list(dev1)
        dev2=list(dev2)
        dev3=list(dev3)
        train1[:],train2[:],train3[:]=zip(*train)
        print 'train-len',len(train1),'dev-len:',len(dev1)

    else:
        print 'no corpus'
        exit()

    train_x1, train_y1 = rep1.represent_instances(train1, max_len=s['max_len_token'])
    dev_x1, dev_y1 = rep1.represent_instances(dev1,max_len=s['max_len_token'])
    test_x1, test_y1 = rep1.represent_instances(test1, max_len=s['max_len_token'])

    train_x2, train_y2 = rep2.represent_instances(train2, max_len=s['max_len_abner'])
    dev_x2, dev_y2 = rep2.represent_instances(dev2,max_len=s['max_len_abner'])
    test_x2, test_y2 = rep2.represent_instances(test2, max_len=s['max_len_abner'])

    train_x3, train_y3 = rep3.represent_instances(train3, max_len=s['max_len_cui'])
    dev_x3, dev_y3 = rep3.represent_instances(dev3,max_len=s['max_len_cui'])
    test_x3, test_y3 = rep3.represent_instances(test3, max_len=s['max_len_cui'])

    inds = range(train_y1.shape[0])
    numpy.random.shuffle(inds)
    batch_num = len(inds) / s['mini_batch']

    # train with early stopping on validation set
    best_f1 = -numpy.inf    
    iter = 0
    max_f=[0,0,0,0,-1]
    last_test_f=[0,0,0,0,-1]
    max_res=0
    each_print=s['each_print']
    model_save=s['model_save']
    max_minibatch=0

    result_file='./ppiac-results/bilstm-token-1l-100d.ppiac2_result'
    model_file='./ppiac-model/bilstm-token-1l-100d.ppiac2_model'

    for epoch in xrange(s['epochs']):
        for minibatch in range(batch_num):
            '''
            model.train_on_batch( [
                                   train_x1[inds[minibatch::batch_num]],
                                   train_x2[inds[minibatch::batch_num]],
                                   train_x3[inds[minibatch::batch_num]]
                                 ],
                                  train_y1[inds[minibatch::batch_num]])
            '''
            model.train_on_batch( train_x1[inds[minibatch::batch_num]], train_y1[inds[minibatch::batch_num]])
            if each_print==True:
                if minibatch % 1 ==0:
                    dev_res=model.predict(dev_x1,batch_size=200)
                    F=Eval.eval_mulclass(dev_y1, dev_res,False, True)
                    if F[2]>max_f[2]:
                        test_res=model.predict(test_x1,batch_size=200)
                        test_F=Eval.eval_mulclass(test_y1, test_res,False,True)
                        max_f[0]=F[0]
                        max_f[1]=F[1]
                        max_f[2]=F[2]
                        max_f[3]=F[3]
                        max_f[4]=epoch
                        last_test_f[0]=test_F[0]
                        last_test_f[1]=test_F[1]
                        last_test_f[2]=test_F[2]
                        last_test_f[3]=test_F[3]
                        max_res=test_res
                        max_minibatch=minibatch
                        if model_save==True:
                            FileUtil.writeFloatMatrix(max_res, result_file)
                            model.save_weights(model_file)

        dev_res=model.predict(dev_x1,batch_size=200)
        F=Eval.eval_mulclass(dev_y1, dev_res,True, True)
        if F[2]>max_f[2]:
            test_res=model.predict(test_x1,batch_size=200)
            test_F=Eval.eval_mulclass(test_y1, test_res,False, True)
            max_f[0]=F[0]
            max_f[1]=F[1]
            max_f[2]=F[2]
            max_f[3]=F[3]
            max_f[4]=epoch
            last_test_f[0]=test_F[0]
            last_test_f[1]=test_F[1]
            last_test_f[2]=test_F[2]
            last_test_f[3]=test_F[3]
            max_res=test_res
            if model_save==True:
                FileUtil.writeFloatMatrix(max_res, result_file)
                model.save_weights(model_file)
        print 'Dev Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d ' % (max_f[0],max_f[1],max_f[2],max_f[3],max_f[4],max_minibatch,epoch)
        print 'Test P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d' % (last_test_f[0],last_test_f[1],last_test_f[2],last_test_f[3],max_f[4],max_minibatch,epoch)
        print '****************************************************************************'
    print model_file,'done' 
    


        
       

