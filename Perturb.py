# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:35:34 2020

@author: Harry
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow import optimizers
#from tqdm import tqdm
import numpy as np
from numpy import float32
import matplotlib.pyplot as plt
import os
#import gensim
#import jieba
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Activation
from tensorflow.keras.datasets import imdb
from tensorflow.python.framework import graph_util
import pandas as pd
import keras_metrics as km
#print(tf.test.is_gpu_available())
#from keras_self_attention import SeqSelfAttention
import gensim
#import pickle
#import seaborn as sns
#import numba

#计算扰动的正则项
def calculate_output(input_sentences, network):
    input_size = len(input_sentences)
    regularization_list = []
    for n in range(input_size):
        sentence = input_sentences[n]
        sentence_representation = network(sentence)
        regularization_list.append(sentence_representation.numpy())
    regularization_list = np.array(regularization_list)
    return np.std(regularization_list, axis=0)
'''
class Perturb(tf.keras.Model):
    def __init__(self, input_sentence, network, rate=0.1, regularization=None, words=None):
        super(Perturb, self).__init__()
        self.input_sentence=input_sentence
        self.sentences_num=tf.shape(input_sentence)[0]
        self.size = tf.shape(input_sentence)[1]#句子中的单词数
        self.dimension = tf.shape(input_sentence)[2]#每一个单词的维度
        self.x=input_sentence#句子的表示
        self.ratio=tf.Variable(tf.random.normal([self.sentences_num,self.size,1],stddev=0.01,seed=1),dtype=float32)#扰动本身的分布，与句子的size相同
        self.scale = tf.constant(float(np.std(np.array(input_sentence))*10))#扰动的范围最大值
        print(self.scale)
        self.rate = rate#平衡最大熵和最小距离的超参数
        self.network = network#文本分类网络
        self.regular = regularization
        if self.regular is not None:
            self.regular = tf.constant(self.regular)
        self.words = words
        if self.words is not None:
            assert self.size == len(words), 'the length of x should be of the same with the lengh of words'  
    #前向传播获取损失
    def call(self):
        ratios = tf.sigmoid(self.ratio)  # batch * S * 1
        x = self.input_sentence + 0.    # S * D
        #print(tf.random.normal(self.size, self.dimension))
        x_noise = x + ratios * tf.random.normal([self.sentences_num,self.size, self.dimension]) * self.scale#加入扰动
        y = self.network(x)  # D or S * D
        y_noise = self.network(x_noise)
        loss = tf.square(y_noise - y)
        #计算扰动后输出与原输出的距离
        if self.regular is not None:
            loss = tf.reduce_mean(loss / tf.square(self.regular))
        else:
            loss = tf.reduce_mean(loss) / tf.reduce_mean(tf.square(y))
        #是扰动满足熵最大原则
        loss=loss - tf.reduce_mean(tf.math.log(ratios)) * self.rate
        #print(loss)
        return loss
    #获取梯度    
    def get_grad(self):
        with tf.GradientTape() as tape:
            g = tape.gradient(self.call(), self.variables)
        return g
    #Adam优化
    def network_learn(self):
        g = self.get_grad()
        optimizers.Adam(lr=0.01).apply_gradients(zip(g, self.variables))
    #获取扰动半径
    def get_sigma(self):
        ratios = tf.sigmoid(self.ratio)  # S * 1
        #print(ratios.numpy()[:,0] *self.scale)
        return ratios.numpy()[:,:,0] *self.scale
    #可视化
    def get_attentions(self):
        return tf.nn.softmax(tf.math.reciprocal(tf.nn.softmax(self.get_sigma())))
    def att_visualize(self):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_attentions()
        _, ax = plt.subplots()
        ax.imshow([sigma_], cmap='GnBu_r')
        ax.set_xticks(range(self.size))
        #ax.set_xticklabels(self.words)
        ax.set_yticks([0])
        ax.set_yticklabels([''])
        plt.tight_layout()
        plt.show()
    def visualize(self):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_sigma()        
        for item in sigma_:
            _, ax = plt.subplots()
            ax.imshow([item], cmap='GnBu_r')
            ax.set_xticks(range(self.size))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.show()
            '''
class Perturb(tf.keras.Model):
    def __init__(self, input_sentences, input_label, network_1, network_2, rate1=0.1, rate2=1.0, regularization=None, words=None):#0.1
        super(Perturb, self).__init__()
        self.input_sentences=input_sentences
        self.sentences_num=tf.shape(input_sentences)[0]
        self.size = tf.shape(input_sentences)[1]#句子中的单词数
        self.dimension = tf.shape(input_sentences)[2]#每一个单词的维度
        self.x=input_sentences#句子的表示
        self.label=input_label
        self.ratio=tf.Variable(tf.random.normal([self.sentences_num, self.size, 1],stddev=0.01),dtype=float32)#扰动本身的分布，与句子的size相同
        #self.ratio=tf.Variable(tf.random.normal([self.size, 1],stddev=0.01),dtype=float32)
        self.scale = tf.constant(float(np.std(np.array(input_sentences))*10))#*10
        self.rate1 = rate1#平衡最大熵和最小距离的超参数
        self.rate2 = rate2#平衡最大熵和最小距离的超参数
        self.network_1 = network_1#文本分类网络
        self.network_2 = network_2#文本分类网络
        self.regular = regularization
        if self.regular is not None:
            self.regular = tf.constant(self.regular)
        self.words = words
        if self.words is not None:
            assert self.size == len(words), 'the length of x should be of the same with the lengh of words'  
    #前向传播获取损失
    def call(self):
        ratios = tf.sigmoid(self.ratio)  # batch * S * 1
        x = self.input_sentences + 0.    # batch * S * D
        x_noise = x + ratios * tf.random.normal([self.sentences_num, self.size, self.dimension],stddev=1.0) * self.scale#加入扰动1.0
        #x_noise = x + ratios * tf.random.normal([self.size, self.dimension],stddev=1.0) * self.scale
        y = self.network_1(x)  # D or S * D
        y_noise = self.network_1(x_noise)  # D or S * D
        yl = self.network_2(x)  # D or S * D
        y_noisel = self.network_2(x_noise)
        loss = tf.square(y_noise - y)
        lossl = tf.square(y_noisel - yl)
        
        #计算扰动后输出与原输出的距离
        if self.regular is not None:
            loss = tf.reduce_mean(loss / tf.square(self.regular))
        else:
            loss = tf.reduce_mean(loss) / tf.reduce_mean(tf.square(y))
            lossl = tf.reduce_mean(lossl) / tf.reduce_mean(tf.square(yl))
        #是扰动满足熵最大原则
        '''
        print(loss)
        print(lossl * self.rate2)
        print(tf.reduce_mean(tf.math.log(ratios)) * self.rate1)
        '''
        loss = loss + lossl * self.rate2 - tf.reduce_mean(tf.math.log(ratios)) * self.rate1
        print(loss)
        return loss
    #获取梯度    
    def get_grad(self):
        with tf.GradientTape() as tape:
            g = tape.gradient(self.call(), self.variables)
        return g
    #Adam优化
    def network_learn(self):
        g = self.get_grad()
        optimizers.Adam(lr=0.01).apply_gradients(zip(g, self.variables))
    #获取扰动半径
    def get_sigma(self):
        ratios = tf.sigmoid(self.ratio)  # S * 1
        #print(ratios.numpy()[:,0] *self.scale)
        return ratios.numpy()[:,:,0] *self.scale
    def get_attention(self):
        ratios = tf.sigmoid(self.ratio)  # S * 1
        sigma_ = ratios.numpy()[:,:,0] *self.scale
        attention=tf.nn.softmax(1-sigma_ / tf.tile(tf.expand_dims(tf.reduce_max(sigma_, axis=-1),1),[1,180]),axis=-1)
        return attention
    #可视化
    def visualize(self):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_sigma()
        #print(sigma_)
        for i in range(sigma_.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([sigma_[i]], cmap='GnBu_r')
            #ax.set_xticks(range(self.size))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.rcParams['figure.figsize']=(24.0,12.0)
            plt.show()
    def attention_visualize_1(self):
        """ Visualize the information loss of every word.
        """    
        attention=tf.nn.softmax(tf.math.reciprocal(self.get_sigma()))
        #print(attention)
        for i in range(attention.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([attention[i]], cmap='GnBu')
            ax.set_xticks(range(self.size))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.show()
    def attention_visualize_2(self):
        """ Visualize the information loss of every word.
        """    
        attention=tf.nn.softmax(tf.math.reciprocal(tf.nn.softmax(self.get_sigma())))
        #print(attention)
        for i in range(attention.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([attention[i]], cmap='GnBu')
            ax.set_xticks(range(self.size))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.show()
    def attention_visualize(self):
        """ Visualize the information loss of every word.
        """    
        attention=self.get_attention()
        #print(attention)
        for i in range(attention.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([attention[i]], cmap='GnBu')
            ax.set_xticks(range(self.size))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.show()






class LSTM_Classification(tf.keras.Model):
    def __init__(self):
        super(LSTM_Classification, self).__init__()
        self.embedding=keras.layers.Embedding(input_dim=20000, output_dim=64, input_length=20)
        self.lstm=keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))
        self.GlobalMaxPooling1D_layer=keras.layers.GlobalMaxPooling1D()
        self.Dense_layer=keras.layers.Dense(units=2,activation='softmax')
    def call(self, inputs):
        x=self.embedding(inputs)
        x=self.lstm(x)       
        x=self.GlobalMaxPooling1D_layer(x)
        x=self.Dense_layer(x)
        return x
    def get_embedding(self, inputs):
        x=self.embedding(inputs)
        return x
    def embedding_to_result(self, inputs):  
        x=self.lstm(inputs)
        x=self.GlobalMaxPooling1D_layer(x)
        return x
    


class LSTM_attention_Classification(tf.keras.Model):
    def __init__(self):
        super(LSTM_attention_Classification, self).__init__()
        self.embedding=keras.layers.Embedding(input_dim=20000, output_dim=64, input_length=15)
        self.lstm=keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))
        self.GlobalMaxPooling1D_layer=keras.layers.GlobalMaxPooling1D()
        self.attention_vector = Dense(128, use_bias=False, activation='tanh')
        self.Dense_layer=keras.layers.Dense(units=2,activation='softmax')
    def call(self, inputs):
        x=self.embedding(inputs)
        x=self.lstm(x)
        ht=self.GlobalMaxPooling1D_layer(x)
        score = dot([x, ht], [2, 1])
        attention_weights=Activation('softmax')(score)
        context_vector = dot([x, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, ht])
        x = self.attention_vector(pre_activation)
        x=self.Dense_layer(x)
        return x
    def get_attentions(self, inputs):
        x=self.embedding(inputs)
        x=self.lstm(x)
        ht=self.GlobalMaxPooling1D_layer(x)
        score = dot([x, ht], [2, 1])
        attention_weights=Activation('softmax')(score)
        return attention_weights
    def get_embedding(self, inputs):
        x=self.embedding(inputs)
        return x
    def embedding_to_result(self, inputs):
        x=self.lstm(inputs)
        ht=self.GlobalMaxPooling1D_layer(x)
        score = dot([x, ht], [2, 1])
        attention_weights=Activation('softmax')(score)
        context_vector = dot([x, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, ht])
        x = self.attention_vector(pre_activation)
        x=self.Dense_layer(x)
        return x
    def embedding_to_h(self, inputs):
        x=self.lstm(inputs)
        #print(inputs)
        ht=self.GlobalMaxPooling1D_layer(x)
        return ht
    def visualize(self,inputs,name,words):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_attentions(inputs)        
        for i in range(sigma_.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([sigma_[i][:]], cmap='GnBu')
            ax.set_xticks(range(20))
            ax.set_xticklabels(words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            #plt.tick_params(labelsize=7)
            plt.rcParams['figure.figsize']=(16.0,8.0)
            plt.savefig('visual/'+name+'.png')



class LSTM_perturb_Classification(tf.keras.Model):
    def __init__(self):
        super(LSTM_perturb_Classification, self).__init__()
        self.embedding=keras.layers.Embedding(input_dim=20000, output_dim=64, input_length=15)
        self.WQ=keras.layers.Dense(units=64, use_bias=False)
        self.Wk=keras.layers.Dense(units=64, use_bias=False)
        self.WV=keras.layers.Dense(units=64, use_bias=False)
        self.attention_vector=keras.layers.Dense(1, use_bias=False, activation='tanh')
        self.lstm=keras.layers.Bidirectional(keras.layers.LSTM(units=128))
        #self.GlobalMaxPooling1D_layer=keras.layers.GlobalMaxPooling1D()
        self.Dense_layer=keras.layers.Dense(units=2,activation='softmax')
    def call(self, inputs):
        x=self.embedding(inputs)
        WQ = self.WQ(x)
        WK = self.Wk(x)
        WV = self.WV(x)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)
        QK = tf.tile(attention_weights,[1,1,64])
        V=QK*WV
        x=self.lstm(V)
        x = self.Dense_layer(x)
        return [x,attention_weights[:,:,0]]
    def get_attentions(self, inputs):
        x=self.embedding(inputs)
        WQ = self.WQ(x)
        WK = self.Wk(x)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)[:,:,0]
        return attention_weights
    def get_embedding(self, inputs):
        x=self.embedding(inputs)
        return x
    def embedding_to_result(self, inputs):
        WQ = self.WQ(inputs)
        WK = self.Wk(inputs)
        WV = self.WV(inputs)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)
        QK = tf.tile(attention_weights,[1,1,64])
        V=QK*WV
        x=self.lstm(V)
        #ht=self.GlobalMaxPooling1D_layer(x)
        x = self.Dense_layer(x)
        return x
    def embedding_to_h(self, inputs):
        WQ = self.WQ(inputs)
        WK = self.Wk(inputs)
        WV = self.WV(inputs)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)
        QK = tf.tile(attention_weights,[1,1,64])
        V=QK*WV
        return V
    def visualize(self,inputs):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_attentions(inputs)
        print(sigma_)
        for i in range(sigma_.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([sigma_[i]], cmap='GnBu')
            #ax.set_xticks(range(self.size))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.rcParams['figure.figsize']=(24.0,12.0)
            plt.show()
class LSTM_test_Classification(tf.keras.Model):
    def __init__(self):
        super(LSTM_test_Classification, self).__init__()
        self.embedding=keras.layers.Embedding(input_dim=20000, output_dim=64, input_length=20)
        self.WQ=keras.layers.Dense(units=64, use_bias=False)
        self.Wk=keras.layers.Dense(units=64, use_bias=False)
        self.WV=keras.layers.Dense(units=64, use_bias=False)
        
        self.attention_vector=keras.layers.Dense(1, use_bias=False, activation='tanh')
        self.lstm=keras.layers.Bidirectional(keras.layers.LSTM(units=128))
        #self.GlobalMaxPooling1D_layer=keras.layers.GlobalMaxPooling1D()
        self.Dense_layer=keras.layers.Dense(units=2,activation='softmax')
    def call(self, inputs):
        x=self.embedding(inputs)
        WQ = self.WQ(x)
        WK = self.Wk(x)
        WV = self.WV(x)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)
        QK = tf.tile(attention_weights,[1,1,64])
        V=QK*WV
        x=self.lstm(V)
        x = self.Dense_layer(x)
        return x
    def get_attentions(self, inputs):
        x=self.embedding(inputs)
        WQ = self.WQ(x)
        WK = self.Wk(x)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)[:,:,0]
        return attention_weights
    def get_embedding(self, inputs):
        x=self.embedding(inputs)
        return x
    def embedding_to_result(self, inputs):
        WQ = self.WQ(inputs)
        WK = self.Wk(inputs)
        WV = self.WV(inputs)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)
        QK = tf.tile(attention_weights,[1,1,64])
        V=QK*WV
        x=self.lstm(V)
        #ht=self.GlobalMaxPooling1D_layer(x)
        x = self.Dense_layer(x)
        #print(x)
        return x
    def embedding_to_h(self, inputs):
        WQ = self.WQ(inputs)
        WK = self.Wk(inputs)
        WV = self.WV(inputs)
        attention_weights = keras.backend.softmax(self.attention_vector(keras.backend.batch_dot(WQ,keras.backend.permute_dimensions(WK, [0, 2, 1]))),axis=1)
        QK = tf.tile(attention_weights,[1,1,64])
        V=QK*WV
        return V
    def visualize(self,inputs):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_attentions(inputs)        
        for i in range(sigma_.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([sigma_[i][:]], cmap='GnBu')
            ax.set_xticks(range(30))
            #ax.set_xticklabels(words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            #plt.tick_params(labelsize=7)
            plt.rcParams['figure.figsize']=(16.0,8.0)
            #plt.savefig('visual/'+name+'.png')
            




#imdb

max_features = 20000
maxlen = 180
batch_size = 32
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()
index_word = {v:k for k,v in word_index.items()}
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
shuffle_ix = np.random.permutation(np.arange(len(x_train)))
x_train = x_train[shuffle_ix,:]
y_train = y_train[shuffle_ix,:]



#sst1
'''
x_train = []
x_test = []
y_train = []
y_test = []
data=pd.read_csv("text-classification/sst1/train.csv")
x_trains=data['x'].tolist()
y_trains=data['y'].tolist()
for item in x_trains:
    newrow=[]
    for num in item.strip().split()[:-1]:
        newrow.append(int(num))
    x_train.append(newrow)
data=pd.read_csv("text-classification/sst1/test.csv")
x_tests=data['x'].tolist()
y_tests=data['y'].tolist()
for item in x_tests:
    newrow=[]
    for num in item.strip().split()[:-1]:
        newrow.append(int(num))
    x_test.append(newrow)

max_features = 20000
maxlen = 20
batch_size = 32
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = keras.utils.to_categorical(y_trains, num_classes=5)
y_test = keras.utils.to_categorical(y_tests, num_classes=5)
'''
#sst2
'''
x_train = []
x_test = []
y_train = []
y_test = []
data=pd.read_csv("text-classification/sst2/tiny/train.csv")
x_trains=data['x'].tolist()
y_trains=data['y'].tolist()
for item in x_trains:
    newrow=[]
    for num in item.strip().split()[:-1]:
        newrow.append(int(num))
    x_train.append(newrow)
data=pd.read_csv("text-classification/sst2/tiny/test.csv")
x_tests=data['x'].tolist()
y_tests=data['y'].tolist()
for item in x_tests:
    newrow=[]
    for num in item.strip().split()[:-1]:
        newrow.append(int(num))
    x_test.append(newrow)

max_features = 20000
maxlen = 30
batch_size = 32
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = keras.utils.to_categorical(y_trains, num_classes=2)
y_test = keras.utils.to_categorical(y_tests, num_classes=2)
import pandas as pd
f1=open("text-classification/sst2/train.txt",'r',encoding='utf-8')
f2=open("text-classification/sst2/test.txt",'r',encoding='utf-8')
dic={}
lines=f1.readlines()
for line in lines:
    words=line[2:].lower().split()
    for word in words:         
        if word not in dic:
            dic[word] = 1
        else:
            dic[word] = dic[word] + 1
lines=f2.readlines()
for line in lines:
    words=line[2:].lower().split()
    for word in words:         
        if word not in dic:
            dic[word] = 1
        else:
            dic[word] = dic[word] + 1
f1.close()
f2.close()
wordlist=sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
newmap={}
i=0
while i<len(wordlist):
    newmap[wordlist[i][0]]=i+1
    i=i+1
index_word = {v:k for k,v in newmap.items()}
index_word[0]='*'
#print(index_word)
#shuffle_ix = np.random.permutation(np.arange(len(x_train)))
#x_train = x_train[shuffle_ix,:]
#y_train = y_train[shuffle_ix,:]
'''
#google word2vector
'''
print("loading word2vec")
Word2VecModel = gensim.models.KeyedVectors.load_word2vec_format('google/GoogleNews-vectors-negative300.bin',binary=True)
vocab_list = [word for word, Vocab in Word2VecModel.key_to_index.items()]
word_index = {" ": 0}
word_vector = {}
embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
'''

'''
amodel=LSTM_test_Classification()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
amodel.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_test, y_test))
#amodel.call(tf.constant(x_train[0:5]))
amodel.save_weights('sst2/epoch/lstm/origin/amodel1')
amodel.save_weights('sst2/epoch/lstm/origin/amodel2')
'''
'''
amodel=LSTM_test_Classification()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
amodel.load_weights('sst2/epoch/lstm/origin/amodel1')
#amodel.embedding_to_h(amodel.get_embedding(tf.constant(x_train[i:i+10])))
#amodel.embedding_to_result(amodel.get_embedding(tf.constant(x_train[i:i+10])))

pmodel=Perturb(amodel.get_embedding(tf.constant(x_train[10:15])),
               amodel.get_embedding(tf.constant(x_train[10:15])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=10.0,rate2=1.0)
attentions=amodel.get_attentions(x_train[10:15])
for j in range(500):
    pmodel.network_learn()
sigma=pmodel.get_sigma()

plt.figure(1, figsize=(60.0,30.0), dpi=300)

ax1 = plt.subplot(611)
ax1.imshow([attentions[1]], cmap='OrRd')
ax1.set_xticks(range(30))
ax1.set_yticks([0])
ax1.set_yticklabels([''])
plt.title("original attention for sentence 1", fontsize=60, loc='center' ,color='blue')

ax2 = plt.subplot(612)
ax2.imshow([sigma[1]], cmap='OrRd_r')
ax2.set_xticks(range(30))
ax2.set_yticks([0])
ax2.set_yticklabels([''])
plt.title("importance distribution for sentence 1", fontsize=60, loc='center' ,color='red')

ax3 = plt.subplot(613)
ax3.imshow([attentions[2]], cmap='OrRd')
ax3.set_xticks(range(30))
ax3.set_yticks([0])
ax3.set_yticklabels([''])
plt.title("original attention for sentence 2", fontsize=60, loc='center' ,color='blue')

ax4 = plt.subplot(614)
ax4.imshow([sigma[2]], cmap='OrRd_r')
ax4.set_xticks(range(30))
ax4.set_yticks([0])
ax4.set_yticklabels([''])
plt.title("importance distribution for sentence 2", fontsize=60, loc='center' ,color='red')

ax5 = plt.subplot(615)
ax5.imshow([attentions[4]], cmap='OrRd')
ax5.set_xticks(range(30))
ax5.set_yticks([0])
ax5.set_yticklabels([''])
plt.title("original attention for sentence 3", fontsize=60, loc='center' ,color='blue')

ax6 = plt.subplot(616)
ax6.imshow([sigma[4]], cmap='OrRd_r')
ax6.set_xticks(range(30))
ax6.set_yticks([0])
ax6.set_yticklabels([''])
plt.title("importance distribution for sentence 3", fontsize=60, loc='center' ,color='red')
'''
'''
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths, load_trained_model_from_checkpoint, Tokenizer, load_vocabulary
paths = get_checkpoint_paths('bert_128')
from tqdm import tqdm
tokenizer=Tokenizer(load_vocabulary(paths.vocab))
def load_data(path):
    global tokenizer
    indices, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name), 'r') as reader:
                  text = reader.read()
            ids, segments = tokenizer.encode(text, max_len=100)
            indices.append(ids)
            sentiments.append(sentiment)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % 32
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    return [indices, np.zeros_like(indices)], np.array(sentiments)
  
  
train_path = os.path.join(os.path.dirname("text-classification/imdb/aclImdb"), 'aclImdb', 'train')
test_path = os.path.join(os.path.dirname("text-classification/imdb/aclImdb"), 'aclImdb', 'test')
train_x, train_y = load_data(train_path)
test_x, test_y = load_data(test_path)
model = load_trained_model_from_checkpoint(
    config_file=paths.config,
    checkpoint_file=paths.checkpoint,
    training=True,
    trainable=True,
    seq_len=100)
for l in model.layers:
    l.trainable = True
class LSTM_bert(tf.keras.Model):
    def __init__(self):
        super(LSTM_bert, self).__init__()
        self.input_layer = model.inputs[:2]
        self.lstm=keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))
        self.GlobalMaxPooling1D_layer=keras.layers.GlobalMaxPooling1D()
        self.attention_vector = Dense(128, use_bias=False, activation='tanh')
        self.Dense_layer=keras.layers.Dense(units=2,activation='softmax')
    def call(self, inputs):
        x=self.input_layer(inputs)
        x=self.lstm(x)
        ht=self.GlobalMaxPooling1D_layer(x)
        score = dot([x, ht], [2, 1])
        attention_weights=Activation('softmax')(score)
        context_vector = dot([x, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, ht])
        x = self.attention_vector(pre_activation)
        x = self.Dense_layer(x)
        return x
    def get_attentions(self, inputs):
        x=self.embedding(inputs)
        x=self.lstm(x)
        ht=self.GlobalMaxPooling1D_layer(x)
        score = dot([x, ht], [2, 1])
        attention_weights=Activation('softmax')(score)
        return attention_weights
    def get_embedding(self, inputs):
        x=self.embedding(inputs)
        return x
    def embedding_to_result(self, inputs):
        x=self.lstm(inputs)
        ht=self.GlobalMaxPooling1D_layer(x)
        return ht
    def visualize(self,inputs):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_attentions(inputs)        
        for i in range(sigma_.shape[0]):
            _, ax = plt.subplots()
            ax.imshow([sigma_[i][:]], cmap='GnBu')
            ax.set_xticks(range(20))
            #ax.set_xticklabels(self.words)
            ax.set_yticks([0])
            ax.set_yticklabels([''])
            plt.tight_layout()
            plt.show()


amodel=LSTM_bert()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
#amodel.call(x_train[0:5])
amodel.fit(x_train, y_train, batch_size=128, epochs=3)


token_dict = load_vocabulary(paths.vocab)
model = load_trained_model_from_checkpoint(
    config_file=paths.config,
    checkpoint_file=paths.checkpoint,
    training=False,
    trainable=True,
    seq_len=100)

tokenizer=Tokenizer(load_vocabulary(paths.vocab))
def get_bert_embdeddings(inputs):
    res=[]
    for item in inputs:
        indices, segments = tokenizer.encode(first=inputs, max_len=100)
        res=res.append(model.predict([np.array([indices]), np.array([segments])])[0])
    return np.array(res)
     

print(tokenizer.tokenize("hello it is me my name is apple and who are you stupid boy"))

print(indices)
print(segments)
print(model.predict([np.array([indices]), np.array([segments])]))
predicts = model.predict([np.array([indices]), np.array([segments])])[0]

model_path = 'bert_128'
embeddings = extract_embeddings(model_path, [mystr])
print(len(embeddings[0]))
'''

'''
amodel=LSTM_test_Classification()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
amodel.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))
amodel.save_weights('sst2/epoch/lstm/origin/amodel1')
amodel.save_weights('sst2/epoch/lstm/origin/amodel2')
'''



#sst2 get perturb
'''
amodel=LSTM_test_Classification()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
amodel.load_weights("sst2/epoch/lstm/origin/amodel2")

ii=0
while (ii+1000)<=6000:
    pmodel=Perturb(amodel.get_embedding(tf.constant(x_train[ii:ii+1000])),
               amodel.get_embedding(tf.constant(y_train[ii:ii+1000])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.1,rate2=0.1)
    for j in range(1000):
        pmodel.network_learn()
    pmodel.save_weights("sst2/epoch/1/train/"+str(ii))
    ii=ii+1000
pmodel=Perturb(amodel.get_embedding(tf.constant(x_train[ii:ii+920])),
               amodel.get_embedding(tf.constant(y_train[ii:ii+920])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.1,rate2=0.1)
for j in range(1000):
    pmodel.network_learn()
pmodel.save_weights("sst2/epoch/1/train/"+str(ii))


i=0
while (i+1000)<=1000:
    pmodel=Perturb(amodel.get_embedding(tf.constant(x_test[i:i+1000])),
               amodel.get_embedding(tf.constant(y_test[i:i+1000])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.1,rate2=0.1)
    for j in range(1000):
        pmodel.network_learn()
    pmodel.save_weights("sst2/epoch/1/test/"+str(i))
    i=i+1000
pmodel=Perturb(amodel.get_embedding(tf.constant(x_test[i:i+821])),
               amodel.get_embedding(tf.constant(y_test[i:i+821])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.1,rate2=0.1)
for j in range(1000):
    pmodel.network_learn()
pmodel.save_weights("sst2/epoch/1/test/"+str(i))
'''

#sst2 train
'''
amodel1=LSTM_perturb_Classification()
amodel1.compile(loss=['categorical_crossentropy'], optimizer="adam", metrics=["categorical_accuracy",km.f1_score()],loss_weights=[1., 0.0])
amodel1.load_weights("sst2/epoch/lstm/origin/amodel1")

amodel2=LSTM_perturb_Classification()
amodel2.compile(loss=['categorical_crossentropy','kullback_leibler_divergence'], optimizer="adam", metrics=["categorical_accuracy",km.f1_score()],loss_weights=[1., 100000.0])
amodel2.load_weights("sst2/epoch/lstm/origin/amodel2")

i=0
pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_train[i:i+1000])),
               amodel2.get_embedding(tf.constant(y_train[i:i+1000])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel1.load_weights("sst2/epoch/1/train/"+str(i))
all_attention1=pmodel1.get_attention()
pmodel2=Perturb(amodel2.get_embedding(tf.constant(x_test[i:i+1000])),
               amodel2.get_embedding(tf.constant(y_test[i:i+1000])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel2.load_weights("sst2/epoch/1/test/"+str(i))
all_attention2=pmodel2.get_attention()

i=1000
pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_train[i:i+1000])),
               amodel2.get_embedding(tf.constant(y_train[i:i+1000])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel1.load_weights("sst2/epoch/1/train/"+str(i))
pmodel2=Perturb(amodel2.get_embedding(tf.constant(x_test[i:i+821])),
               amodel2.get_embedding(tf.constant(y_test[i:i+821])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel2.load_weights("sst2/epoch/1/test/"+str(i))

all_attention1=tf.concat([all_attention1,pmodel1.get_attention()],axis=0)
all_attention2=tf.concat([all_attention2,pmodel2.get_attention()],axis=0)

i=2000
pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_train[i:i+1000])),
               amodel2.get_embedding(tf.constant(y_train[i:i+1000])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel1.load_weights("sst2/epoch/1/train/"+str(i))
#pmodel2=Perturb(amodel2.get_embedding(tf.constant(x_test[i:i+210])), amodel2.embedding_to_result, rate=0.1)
#pmodel2.load_weights("sst1/epoch/2/test/"+str(i))
all_attention1=tf.concat([all_attention1,pmodel1.get_attention()],axis=0)
#all_attention2=tf.concat([all_attention2,pmodel2.get_attention()],axis=0)

i=3000
while (i+1000)<=6000:
    pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_train[i:i+1000])),
               amodel2.get_embedding(tf.constant(y_train[i:i+1000])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
    pmodel1.load_weights("sst2/epoch/1/train/"+str(i))
    all_attention1=tf.concat([all_attention1,pmodel1.get_attention()],axis=0)
    i=i+1000
pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_train[i:i+920])),
               amodel2.get_embedding(tf.constant(y_train[i:i+920])),
               amodel2.embedding_to_h,amodel2.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel1.load_weights("sst2/epoch/1/train/"+str(i))
all_attention1=tf.concat([all_attention1,pmodel1.get_attention()],axis=0)

all_attention1=all_attention1.numpy()
all_attention2=all_attention2.numpy()

#y_train=np.concatenate((y_train, all_attention1), axis=1)
#y_test=np.concatenate((y_test, all_attention2), axis=1)
#result = amodel.evaluate(x_test, y_test, batch_size=128)
#print(np.shape(y_train))
#print(np.shape(all_attention))
#print(amodel.call(tf.constant([x_train[0]])))
amodel1.fit(x_train, [y_train,all_attention1], batch_size=128, epochs=3, validation_data=(x_test, [y_test,all_attention2]))
#amodel2.fit(x_train, [y_train,all_attention1], batch_size=128, epochs=3, validation_data=(x_test, [y_test,all_attention2]))
amodel1.save_weights("sst2/epoch/lstm/1/amodel1")
#amodel2.save_weights("sst2/epoch/lstm/1/amodel2")
'''



#visual

'''
amodel1=LSTM_test_Classification()
amodel1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel1.load_weights("sst2/epoch/lstm/1/amodel1")
amodel2=LSTM_test_Classification()
amodel2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel2.load_weights("sst2/epoch/lstm/1/amodel2")
i=0
while i<1821:
    if (amodel1.predict(np.array([x_test[i]]))[0][0]>amodel1.predict(np.array([x_test[i]]))[0][1])!=(y_test[i][0]>y_test[i][1]):
        if (amodel2.predict(np.array([x_test[i]]))[0][0]>amodel2.predict(np.array([x_test[i]]))[0][1])==(y_test[i][0]>y_test[i][1]):
            words=[]
            for word in x_test[i]:
                words.append(index_word[word])
            amodel1.visualize(tf.constant([x_test[i]]),str(i)+"-1",words)
            amodel2.visualize(tf.constant([x_test[i]]),str(i)+"-2",words)
            print(i)
    i=i+1
'''



'''
amodel1=LSTM_attention_Classification()
amodel1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",km.f1_score()])
amodel1.load_weights("sst2/epoch/lstm/1/amodel1")
amodel2=LSTM_attention_Classification()
amodel2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",km.f1_score()])
amodel2.load_weights("sst2/epoch/lstm/1/amodel2")

f=open("records.txt",'r')
index=f.readlines()
for i in index:
    words=[]
    for word in x_test[int(i)]:
        words.append(index_word[word])
    amodel1.visualize(tf.constant([x_test[int(i)]]),str(int(i))+"-1",words)
    amodel2.visualize(tf.constant([x_test[int(i)]]),str(int(i))+"-2",words)

amodel1.evaluate(x_test, y_test, batch_size=128)
amodel2.evaluate(x_test, y_test, batch_size=128)
'''


'''
model=LSTM_attention_Classification()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test))

amodel=LSTM_test_Classification()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
amodel.load_weights('sst2/lstm/amodel')
pmodel=Perturb(amodel.get_embedding(tf.constant(x_test[0:1000])), amodel.embedding_to_result, rate=0.1)
pmodel.load_weights("sst2/sst2_test_perturb10/0")
all_attention=pmodel.get_attention()
pmodel=Perturb(amodel.get_embedding(tf.constant(x_test[1000:1821])), amodel.embedding_to_result, rate=0.1)
pmodel.load_weights("sst2/sst2_test_perturb10/1000")
all_attention=tf.concat([all_attention,pmodel.get_attention()],axis=0)
x_test=np.concatenate((x_test, all_attention), axis=1)
result = amodel.evaluate(x_test, y_test, batch_size=128)
'''
'''
amodel1=LSTM_perturb_Classification()
amodel1.compile(loss=mycrossentropy1, optimizer="adam", metrics=["categorical_accuracy"])
amodel1.load_weights('sst2/lstm/amodel')

amodel2=LSTM_perturb_Classification()
amodel2.compile(loss=mycrossentropy2, optimizer="adam", metrics=["categorical_accuracy"])
amodel2.load_weights('sst2/lstm/amodel')


pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_test[0:1000])), amodel2.embedding_to_result, rate=0.1)
pmodel2=Perturb(amodel2.get_embedding(tf.constant(x_test[0:1000])), amodel2.embedding_to_result, rate=0.1)
pmodel1.load_weights("imdb_perturb/perturb_base1000_train_len100/"+str(i))
pmodel2.load_weights("imdb_perturb/perturb_base1000_test_len100/"+str(i))
all_attention1=pmodel1.get_attention()
all_attention2=pmodel2.get_attention()
i=100
while (i+100)<=1000:
    pmodel1=Perturb(amodel2.get_embedding(tf.constant(x_test[i:i+100])), amodel2.embedding_to_result, rate=0.1)
    pmodel1.load_weights("imdb_perturb/perturb_base1000_train_len100/"+str(i))
    all_attention1=tf.concat([all_attention1,pmodel1.get_attention()],axis=0)
    i=i+100
i=100
while (i+100)<=25000:
    pmodel2=Perturb(amodel2.get_embedding(tf.constant(x_test[i:i+100])), amodel2.embedding_to_result, rate=0.1)
    pmodel2.load_weights("imdb_perturb/perturb_base1000_test_len100/"+str(i))
    all_attention2=tf.concat([all_attention2,pmodel2.get_attention()],axis=0)
    #print(i)
    i=i+100
all_attention1=all_attention1.numpy()
all_attention2=all_attention2.numpy()
#print(np.shape(all_attention))
y_train=np.concatenate((y_train, all_attention1), axis=1)
y_test=np.concatenate((y_test, all_attention2), axis=1)
#result = amodel.evaluate(x_test, y_test, batch_size=128)
#print(np.shape(y_train))
#print(np.shape(all_attention))
#print(amodel.call(tf.constant([x_train[0]])))
amodel1.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))
amodel2.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))
'''

#amodel.visualize(tf.constant([x_train[10],x_train[11],x_train[12],x_train[13],x_train[14]]))






'''
max_features = 20000
maxlen = 100
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

amodel1=LSTM_attention_Classification()
amodel1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel1.load_weights("manytimes/lstm/20000_25000/amodel1")
amodel1.evaluate(x_test, y_test, batch_size=128)

amodel2=LSTM_attention_Classification()
amodel2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel2.load_weights("manytimes/lstm/20000_25000/amodel2")
amodel2.evaluate(x_test, y_test, batch_size=128)
'''
'''
amodel1=LSTM_test_Classification()
amodel1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel1.load_weights('sst2/epoch/lstm/1/amodel1')
amodel2=LSTM_test_Classification()
amodel2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel2.load_weights('sst2/epoch/lstm/1/amodel2')
amodel1.evaluate(x_test, y_test, batch_size=128)
amodel2.evaluate(x_test, y_test, batch_size=128)
'''


amodel=LSTM_test_Classification()
amodel.load_weights('imdb/epoch/lstm/origin/amodel1')



pmodel=Perturb(amodel.get_embedding(tf.constant(x_train[0:5])),
               amodel.get_embedding(tf.constant(y_train[0:5])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.0,rate2=0.0)
for i in range(1000):
    pmodel.network_learn()
amodel.visualize(tf.constant(x_train[0:5]))
pmodel.attention_visualize()

'''
amodel=LSTM_attention_Classification()
amodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
amodel.load_weights('imdb/epoch/lstm/origin/amodel2')
i=0

while (i+100)<=25000:
    pmodel1=Perturb(amodel.get_embedding(tf.constant(x_train[i:i+100])),
               amodel.get_embedding(tf.constant(y_train[i:i+100])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.1,rate2=0.01)
    pmodel2=Perturb(amodel.get_embedding(tf.constant(x_test[i:i+100])),
               amodel.get_embedding(tf.constant(y_test[i:i+100])),
               amodel.embedding_to_h,amodel.embedding_to_result,rate1=0.1,rate2=0.01)
    for j in range(500):
        pmodel1.network_learn()
        pmodel2.network_learn()
        pmodel1.save_weights("imdb/epoch/1/train/"+str(i))
        pmodel2.save_weights("imdb/epoch/1/test/"+str(i))
    print(i)
    i=i+100
'''
'''
amodel1=LSTM_perturb_Classification()
amodel1.compile(loss=mycrossentropy1, optimizer="adam", metrics=["categorical_accuracy",km.f1_score()])
amodel1.load_weights('imdb/epoch/lstm/origin/amodel1')
amodel2=LSTM_perturb_Classification()
amodel2.compile(loss=mycrossentropy2, optimizer="adam", metrics=["categorical_accuracy",km.f1_score()])
amodel2.load_weights('imdb/epoch/lstm/origin/amodel1')

i=0
pmodel1=pmodel1=Perturb(amodel1.get_embedding(tf.constant(x_train[i:i+100])),
               amodel1.get_embedding(tf.constant(y_train[i:i+100])),
               amodel1.embedding_to_h,amodel1.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel2=Perturb(amodel1.get_embedding(tf.constant(x_test[i:i+100])),
               amodel1.get_embedding(tf.constant(y_test[i:i+100])),
               amodel1.embedding_to_h,amodel1.embedding_to_result,rate1=0.1,rate2=0.01)
pmodel1.load_weights("imdb/epoch/1/train/"+str(i))
pmodel2.load_weights("imdb/epoch/1/test/"+str(i))
all_attention1=pmodel1.get_attention()
all_attention2=pmodel2.get_attention()
i=100
while (i+100)<=25000:
    pmodel1=pmodel1=Perturb(amodel1.get_embedding(tf.constant(x_train[i:i+100])),
               amodel1.get_embedding(tf.constant(y_train[i:i+100])),
               amodel1.embedding_to_h,amodel1.embedding_to_result,rate1=0.1,rate2=0.01)
    pmodel2=Perturb(amodel1.get_embedding(tf.constant(x_test[i:i+100])),
               amodel1.get_embedding(tf.constant(y_test[i:i+100])),
               amodel1.embedding_to_h,amodel1.embedding_to_result,rate1=0.1,rate2=0.01)
    pmodel1.load_weights("imdb/epoch/1/train/"+str(i))
    pmodel2.load_weights("imdb/epoch/1/test/"+str(i))
    all_attention1=tf.concat([all_attention1,pmodel1.get_attention()],axis=0)
    all_attention2=tf.concat([all_attention2,pmodel2.get_attention()],axis=0)
    i=i+100
all_attention1=all_attention1.numpy()
all_attention2=all_attention2.numpy()
#print(np.shape(all_attention))
y_train=np.concatenate((y_train, all_attention1), axis=1)
y_test=np.concatenate((y_test, all_attention2), axis=1)
#result = amodel.evaluate(x_test, y_test, batch_size=128)
#print(np.shape(y_train))
#print(np.shape(all_attention))
#print(amodel.call(tf.constant([x_train[0]])))
amodel1.fit(x_train, y_train, batch_size=128, epochs=7, validation_data=(x_test, y_test))
amodel2.fit(x_train, y_train, batch_size=128, epochs=7, validation_data=(x_test, y_test))
'''
