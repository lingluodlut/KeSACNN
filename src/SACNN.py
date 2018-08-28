import FileUtil
import numpy
import time
import sys
import os
import random
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(123)
random.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True
from keras import backend as K

tf.set_random_seed(345)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from Represent_luo import RepresentationLayer
import Eval
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Lambda
from keras.layers import Embedding,Input
from keras.layers import LSTM, SimpleRNN,TimeDistributed
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import BatchNormalization
from keras.layers.merge import Concatenate,Average,Add
from keras.layers import *
from attention_keras import *


def conv_block(inputs,
               filters=300,
               window=3,
               strides=1,
               activation='relu',
               namecb='',name_j=0):


    x=Conv1D(filters, window, strides=strides, padding='same',name=namecb+str(name_j))(inputs)
    x=BatchNormalization(name=namecb+'BN'+str(name_j))(x)
    x=Activation(activation)(x)

    x=Conv1D(filters, window, padding='same',name=namecb+str(name_j+1))(x)
    x=BatchNormalization(name=namecb+'BN'+str(name_j+1))(x)

    shortcut=Conv1D(filters, window, strides=strides, padding='same',name=namecb+'short'+str(name_j+1))(inputs)
    shortcut=BatchNormalization(name=namecb+'short_BN'+str(name_j+1))(shortcut)
    #shortcut
    if s['shortcut']==True:
        x=Add()([x,shortcut])
    x=Activation(activation)(x)
    return x

def mulhead_self_att(inputs,
                     head_num=4,
                     add_nor=True,
                     activation='linear'):
    x=inputs
    unit_num=int(x._keras_shape[-1])
    x=Self_Attention(head_num,int(unit_num/head_num))([x,x,x])
    if add_nor==True:
        x=BatchNormalization()(x)
        x=Add()([x,inputs])
    x=Activation(activation)(x)
    return x

if __name__ == '__main__':

    s = {
         'max_len_token':600, 
         'hidden_dims':64,
         'block_num':3,
         'shortcut':True,
         'self_att':True,
         'mini_batch':32,
         'epochs':25,
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
                          weights=[rep1.vec_table], input_length=s['max_len_token'],name='ch1_emb')
    chanel1_vec=chanel1_emb(chanel1_input)

    chanel1_vec=Dropout(0.5)(chanel1_vec)
    chanel1_x=chanel1_vec

    
    num=s['hidden_dims']
    j=1
    chanel1_x=conv_block(inputs=chanel1_x, filters=num, window=3, activation='relu',namecb='ch1_conv',name_j=j)
    chanel1_x=MaxPooling1D(pool_size=3,strides=2)(chanel1_x)

    j=j+2
    num=num*2

    for i in range(s['block_num']-1):
        if s['self_att']==True:
            chanel1_x=mulhead_self_att(inputs=chanel1_x)
            
        chanel1_x=conv_block(inputs=chanel1_x, filters=num, window=3, activation='relu',namecb='ch1_conv',name_j=j)
        chanel1_x=MaxPooling1D(pool_size=3,strides=2)(chanel1_x)

        j=j+2
        num=num*2

    if s['self_att']==True:
        chanel1_x=mulhead_self_att(inputs=chanel1_x)
        
    chanel1_x=conv_block(inputs=chanel1_x, filters=num, window=3, activation='relu',namecb='ch1_conv',name_j=j)
    
#concat    

    concat_x=chanel1_x
    concat_x=GlobalMaxPooling1D()(concat_x)


    dense_hidnum=int(concat_x._keras_shape[-1])
    concat_x=Dense(dense_hidnum,activation='relu',name='new4')(concat_x)
    concat_x=Dense(dense_hidnum/2,activation='relu',name='new5')(concat_x)
    concat_x=Dropout(0.5)(concat_x)
    predict=Dense(2,activation='softmax',name='new2')(concat_x)
    
    model= Model(inputs=[chanel1_input],outputs=predict)

    opt = Adadelta()
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
        print ('train-len',len(train1),'dev-len:',len(dev1))

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
        print ('train-len',len(train1),'dev-len:',len(dev1))

    else:
        print ('no corpus')
        exit()

    train_x1, train_y1 = rep1.represent_instances(train1, max_len=s['max_len_token'])
    dev_x1, dev_y1 = rep1.represent_instances(dev1,max_len=s['max_len_token'])
    test_x1, test_y1 = rep1.represent_instances(test1, max_len=s['max_len_token'])


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

    result_file='./ppiac-results/sacnn-4b-100d.ppiac2_testresult'
    model_file='./ppiac-model/sacnn-4b-100d.ppiac2_model'

    for epoch in xrange(s['epochs']):
        for minibatch in range(batch_num):
            model.train_on_batch( 
                                   train_x1[inds[minibatch::batch_num]],
                                   train_y1[inds[minibatch::batch_num]])
        
            if each_print==True:
                if minibatch % 1 ==0:
                    dev_res=model.predict(dev_x1,batch_size=100)
                    F=Eval.eval_mulclass(dev_y1, dev_res,False, True)
                    if F[2]>max_f[2]:
                        test_res=model.predict(test_x1,batch_size=100)
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

        dev_res=model.predict(dev_x1,batch_size=100)
        F=Eval.eval_mulclass(dev_y1, dev_res,True, True)
        if F[2]>max_f[2]:
            test_res=model.predict(test_x1,batch_size=100)
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
        print ('Dev Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d ' % (max_f[0],max_f[1],max_f[2],max_f[3],max_f[4],max_minibatch,epoch))
        print ('Test P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d' % (last_test_f[0],last_test_f[1],last_test_f[2],last_test_f[3],max_f[4],max_minibatch,epoch))
        print ('****************************************************************************')
    print (model_file,'done') 
    print (s)   
        
       

