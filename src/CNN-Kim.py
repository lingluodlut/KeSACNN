import numpy
import FileUtil
import random
import time
numpy.random.seed(123)
random.seed(12345)

import time
import sys
import subprocess
import os
from Represent_luo import RepresentationLayer

import Eval

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Input
from keras.layers import LSTM, SimpleRNN
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad,Nadam,Adamax
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Average,Add 
from keras import backend as K



if __name__ == '__main__':

    s = {
         'max_len':600, 
         'hidden_dims':150,
         'filter_sizes':[3,4,5],
         'num_filters':100,
         'mini_batch':32,
         'epochs':20,
         'each_print':True,
         'model_save':True,
         'corpus':'PPIAC_BC2'#PPIAC_BC3,PPIAC_BC2
    }
    print s
    folder = '/home/BIO/luoling/PPIAC_CNN/'
    rep1 = RepresentationLayer(folder + 'word2vec_model/bc_ex3_token_100.word2vec_model',vec_size=100, frequency=500000)

    model_input=Input(shape=(s['max_len'],),dtype='int32')
    model_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], \
                        weights=[rep1.vec_table], input_length=s['max_len'],trainable=True)
    model_vec=model_emb(model_input)
    model_vec=Dropout(0.5)(model_vec)

    conv_blocks = []
    for sz in s['filter_sizes']:
        conv = Conv1D(filters=s['num_filters'],kernel_size=sz,padding='valid',activation='relu')(model_vec)
        conv=GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    conv_all = Concatenate()(conv_blocks)

    
    model_fc=Dense(s['hidden_dims'],activation='relu')(conv_all)
    model_fc=Dropout(0.5)(model_fc)

    predict=Dense(2,activation='softmax')(model_fc)

    model=Model(inputs=model_input,outputs=predict)


    opt = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()

    Corpus=s['corpus']

    if Corpus=='PPIAC_BC3':

        test1 = [line.strip() for line in open('/home/BIO/luoling/PPIAC_CNN/corpus/bc3_ppi_test.token.lab')]
        train1 = [line.strip() for line in open('/home/BIO/luoling/PPIAC_CNN/corpus/bc3_dev_ppi_train.token.lab')]

        #10% to dev, remaining to train
        dev_num=int(len(train1)*0.1)
        bc3_dev=train1[0:4000]

        random.shuffle(bc3_dev)
        dev1=bc3_dev[0:dev_num]
        train1=bc3_dev[dev_num:]+train1[4000:]
        random.shuffle(train1)
        print 'BC3-PPIAC   train-len:',len(train1),'dev-len:',len(dev1)

    elif Corpus=='PPIAC_BC2':
        test1 = [line.strip() for line in open('/home/BIO/luoling/PPIAC_CNN/corpus/bc2_ppi_test.token.lab')]
        train1 = [line.strip() for line in open('/home/BIO/luoling/PPIAC_CNN/corpus/bc2_ppi_train.token.lab')]

        #10% to dev, remaining to train
        dev_num=int(len(train1)*0.1)
        random.shuffle(train1)
        dev1= train1[0:dev_num]
        train1=train1[dev_num:]
        print 'BC2-PPIAC   train-len',len(train1),'dev-len:',len(dev1)
    else:
        print 'no corpus'
        exit()
    
    train_x1, train_y1 = rep1.represent_instances(train1, max_len=s['max_len'])
    dev_x1,dev_y1 = rep1.represent_instances(dev1, max_len=s['max_len'])
    test_x1, test_y1 = rep1.represent_instances(test1, max_len=s['max_len'])
#    print train_x1
    inds = range(train_y1.shape[0])
    numpy.random.shuffle(inds)
    batch_num = len(inds) / s['mini_batch']

    # train with early stopping on validation set
    max_f=[0,0,0,0,-1]
    last_test_f=[0,0,0,0,-1]
    max_res=0#
    each_print=s['each_print']
    max_minibatch=0
    model_save=s['model_save']
    model_file='./ppiac-model/yoonkim-cnn-token-100d.ppiac2_model'
    result_file='./ppiac-results/yoonkim-cnn-token-100d.ppiac2_result'


    for epoch in xrange(s['epochs']):

        for minibatch in range(batch_num):
            model.train_on_batch( train_x1[inds[minibatch::batch_num]],
                                  train_y1[inds[minibatch::batch_num]])

            if each_print==True:
                if minibatch % 1==0:
                    dev_res = model.predict(dev_x1,batch_size=200)

                    F=Eval.eval_mulclass(dev_y1, dev_res, False,True)
                    if F[2]>max_f[2]:
                        test_res = model.predict(test_x1,batch_size=200)
                        F_test=Eval.eval_mulclass(test_y1, test_res, False,True)

                        max_f[0]=F[0]
                        max_f[1]=F[1]
                        max_f[2]=F[2]
                        max_f[3]=F[3]
                        max_f[4]=epoch
                        last_test_f[0]=F_test[0]
                        last_test_f[1]=F_test[1]
                        last_test_f[2]=F_test[2]
                        last_test_f[3]=F_test[3]
                        max_res=test_res
                        max_minibatch=minibatch
                        if model_save==True:
                            FileUtil.writeFloatMatrix(max_res, result_file)
                            model.save_weights(model_file)

        dev_res = model.predict(dev_x1,batch_size=200)
        F=Eval.eval_mulclass(dev_y1, dev_res, True,True)

        if F[2]>max_f[2]:
            test_res = model.predict(test_x1,batch_size=200)
            F_test=Eval.eval_mulclass(test_y1, test_res, False,True)

            max_f[0]=F[0]
            max_f[1]=F[1]
            max_f[2]=F[2]
            max_f[3]=F[3]
            max_f[4]=epoch
            last_test_f[0]=F_test[0]
            last_test_f[1]=F_test[1]
            last_test_f[2]=F_test[2]
            last_test_f[3]=F_test[3]
            max_res=test_res
            if model_save==True:
                model.save_weights(model_file)
                FileUtil.writeFloatMatrix(max_res, result_file)

        print 'Dev Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d, time=%.1f ' % (max_f[0],max_f[1],max_f[2],max_f[3],max_f[4],max_minibatch,epoch,end-start)
        print 'Test P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d, time=%.1f ' % (last_test_f[0],last_test_f[1],last_test_f[2],last_test_f[3],max_f[4],max_minibatch,epoch,end-start)
        print '*****************************************************'
    print model_file,'done'        
       

