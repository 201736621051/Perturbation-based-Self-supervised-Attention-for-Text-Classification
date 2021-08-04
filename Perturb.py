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
