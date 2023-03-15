# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'
import numpy as np
from sklearn.model_selection import train_test_split

""" load data """
def Dataset():
         
    data=np.loadtxt("dataset/statlog.txt")
    label=np.loadtxt("dataset/statlog_label.txt")
    
    train,test=train_test_split(data,test_size=0.3)
    train_label,test_label=train_test_split(label,test_size=0.3)
    
    train_labels,train_data = train_label, train        
    test_labels,test_data= test_label , test           

    num_dims = train_data.shape[1] 
    time_steps=1 

    train_data = train_data.reshape((train_data.shape[0],time_steps,num_dims))
    test_data = test_data.reshape((test_data.shape[0],time_steps,num_dims))

    return train_labels,test_labels,time_steps,num_dims,train_data,test_data

