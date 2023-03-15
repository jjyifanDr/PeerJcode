# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'
import tensorflow as tf
import numpy as np
from Function import variable_summaries
import KernelFunction


class config():
	def __init__(self, time_steps, num_dim):
		self.time_steps = time_steps
		self.num_dim = num_dim
		self.num_hidden = 100
		self.keep_prob = 0.98 
		self.lamda = 5.0
		self.gamma = 0.05
		self.tol = 1e-1
		self.num_epochs = 100
               

class aesvm():
    
    def weights(self, shape):
        return tf.compat.v1.get_variable(name="weight",shape=shape,dtype=tf.float32,initializer=tf.random_normal_initializer())
    

    def __init__(self, config,data,_data):

        self.input = tf.compat.v1.placeholder(tf.float32,shape=[None,config.time_steps,config.num_dim])
        self.batch_size = tf.shape(self.input)[0]

        self.cen = tf.compat.v1.get_variable("cen", [config.num_hidden], tf.float32, tf.random_normal_initializer())
        self.ra = tf.compat.v1.get_variable("ra", initializer=tf.constant(0.1))
        
        
        with tf.compat.v1.variable_scope("weights"):
            self.weights = self.weights(shape=[config.time_steps, 1])
            variable_summaries(self.weights)
            tf.compat.v1.summary.histogram('histogram', self.weights)
            
            
        rate=config.keep_prob
        
        
        with tf.compat.v1.variable_scope("encoder"):
            inputs = tf.nn.dropout(self.input, rate)
            inputs = tf.unstack(inputs,config.time_steps,1)
            self.enc_cell = tf.keras.layers.LSTMCell(config.num_hidden)

            enc_ouputs,enc_state = tf.nn.static_rnn(self.enc_cell,inputs,dtype=tf.float32)
            enc_ouputs = tf.stack(enc_ouputs,axis=0) 
            enc_outputs = tf.transpose(enc_ouputs, perm=[1, 2, 0]) 
            op = lambda x: tf.matmul(x,self.weights)
            z = tf.map_fn(op,enc_outputs)
            z = tf.squeeze(z)
            
        
        with tf.compat.v1.variable_scope("estimation_ratio"):
            c1=1.0
            c2=1e-8
            self.distance = tf.map_fn(tf.norm,
				(z - tf.reshape(tf.tile(self.cen,[self.batch_size]),[self.batch_size, config.num_hidden])))
            distance2 = tf.square(self.distance)
            residue = tf.nn.relu(distance2 - tf.square(self.ra))
            penalty = tf.nn.relu(tf.exp(tf.square(self.ra) - distance2) - c1) + c2
            penalty = tf.map_fn(lambda x: tf.divide(x,tf.reduce_sum(penalty)),penalty)
            self.penalty = penalty
            self.label = (self.ra * (config.tol+tf.reduce_min(KernelFunction.CalKernel_(np.array(data)  ) ) ) >= self.distance)  
            

        with tf.compat.v1.variable_scope("decoder"):
            dec_inputs = [tf.zeros([self.batch_size,config.num_dim], dtype=tf.float32) for _ in range(config.time_steps)]
            self.dec_cell = tf.keras.layers.LSTMCell(config.num_hidden)	
            dec_output,dec_state = tf.nn.static_rnn(self.dec_cell,dec_inputs,initial_state=enc_state,dtype=tf.float32)			
            dec_output = tf.transpose(tf.stack(dec_output[::-1]),perm=[1,0,2]) 
            dec_output = tf.layers.dense(inputs=dec_output,units=config.num_dim,activation='sigmoid')
            

        with tf.compat.v1.variable_scope("loss"):

            self.rec_diff = self.input - dec_output 
			
            rec_error = tf.reduce_mean(tf.reduce_mean(tf.pow(self.rec_diff, 2), axis=1), axis=1) 
            penalized_rec_error = tf.reduce_mean(tf.multiply(rec_error, penalty))
			
            _loss = tf.square(self.ra) + config.gamma * tf.reduce_sum(residue) + tf.reduce_mean(distance2)+tf.reduce_mean(tf.norm(KernelFunction.CalKernel_(data),ord=1))+tf.reduce_mean(tf.norm(KernelFunction.CalKernel_(data),ord=2))# add L1,L2 regular 
            self.loss = _loss + config.lamda * penalized_rec_error  