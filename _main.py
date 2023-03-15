# -*- coding: utf-8 -*-

#<--author:zhengjian.002@163.com--->

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import mean_squared_error,accuracy_score,f1_score
from Function import rmse, sensitivity_score
from LoadDatasets import Dataset
import AESVM





     
if __name__ == '__main__':
    
    #loading data..............   
    train_label,test_label,time_step, num_dim, train_data,test_data=Dataset() 
    
    print("\n...................Load Data end..................\n")
    print("\n...................Running start.................\n")
    t1=time.time()
    config = AESVM.config(time_step,num_dim)
    model = AESVM.aesvm(config,np.mat(train_data),np.mat(test_data))


    with tf.name_scope('train'):
	     train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(model.loss)   
         

    with tf.compat.v1.Session () as sess:
        scalar=tf.summary.scalar('weights',model.weights)         
        merged = tf.compat.v1.summary.merge_all(scalar)
        sess.run(tf.compat.v1.global_variables_initializer())


        """ loop epoch """
        for e in range(config.num_epochs):			
        #""" train """
                _,train_loss,train_label_pred,r,d,penalty = sess.run([train_op, model.loss, model.label, model.ra, model.distance, model.penalty],feed_dict={model.input:train_data})	 
                

                if e %10==0:
                        print("\n train epoch <.....{}/{}.....>".format(e,config.num_epochs))
        print ("\n.....Training complete......\n")
        print("Training accurcay = {:.6f}".format(accuracy_score(train_label,train_label_pred)))                                      
		
        
        #""" test """
        test_loss, test_label_pred, test_diff_pred = sess.run([model.loss, model.label, model.rec_diff],feed_dict={model.input:test_data})


           
        t2=time.time()
        times=t2-t1
        
        print (".........[Testing results].........")
        
        print("\n......Accuray.....")
        print("accurcay = {:.6f}".format(accuracy_score(test_label,test_label_pred)))

      
        print("\n......F1-score.....")
        print("[f1-socre] = {:.6f}".format(f1_score(test_label,test_label_pred,average='weighted')))
        
        
        print("\n......Sensitivity......")
        print("{sensitivity} =""%.6f"%(sensitivity_score(test_label,test_label_pred)))
        
            
        print("\n......MSE......")
        print("mse = {:.6f}".format(mean_squared_error(test_label,test_label_pred)))
        
        
        print("......RMSE......")
        print("rmse = {:.6f}".format(rmse(test_label,test_label_pred)))
        
        print("\n.....Running time........")
        print("time = {:.2f}".format(times))
       
            
        print("\n........Running End.........\n")
        

        
#<--author:zhengjian.002@163.com--->


        


  

   

    
    
    
    
    
    
    

        

        