# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'
import numpy as np
import tensorflow as tf

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
 
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)


def rmse(predictions, targets):
	return np.sqrt(np.mean((predictions - targets) ** 2))


# def Regular_Iterm(L, theta):     # using L2-norm or Regular_Iterm
#    return  np.dot(theta, theta) * L 





def sensitivity_score(y_pred, y_true):
    epsilon = 1e-7
    tp = np.sum( (1- y_pred)*(1-y_true), axis=0)
    fn = np.sum((1- (1- y_pred))*(1-y_true), axis=0)
    sen = tp/(tp+fn+epsilon)
 
    return sen





