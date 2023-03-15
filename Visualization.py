# -*- coding: utf-8 -*-
"""

@author: zhengjian.002@163.com
"""


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import exit
import KernelFunction
from LoadDatasets import Dataset



    

def Visualization_kernel( _data, r=0.3, nn=2):
        
        data=_data.numpy()        #a tensor is changed into a array

        # compute the range of grid 
        numGrids = np.rint(data.shape[0]/nn).astype(int)  # number of grids
        x_range = np.zeros(shape=(numGrids, 2))
        for i in range(2):  
            _tmp_ = (np.max(data[:, i])-np.min(data[:, i]))*r
            xlim_1 = np.min(data[:, i])-_tmp_
            xlim_2 = np.max(data[:, i])+_tmp_
            x_range[:, i] = np.linspace(xlim_1, xlim_2, numGrids)
        
        # grid
        xv, yv = np.meshgrid(x_range[:, 0], x_range[:, 1])
        
        num1 = xv.shape[0]
        distance = np.zeros(shape=(num1, num1))
        
                     
        # plot the contour (3D)
        fig = plt.figure(figsize = (20, 6))  
        
        ax = fig.add_subplot(1, 1, 1, projection='3d') 

        ada = ax.plot_surface(-1.0*xv, -1.0*yv, -1.0*distance, cmap=plt.cm.jet)
        ax.contourf(-1.0*xv, -1.0*yv, -1.0*distance, zdir='z', offset=np.min(-1*distance)*0.9, cmap=plt.cm.coolwarm)
        #ax1.set_zlim(np.min(distance)*0.9, np.max(distance)*1.05)
        ax.set_zlim(-1*np.max(distance)*1.05,-1*np.min(distance)*0.9)

        plt.show()
        plt.savefig("./SyntheticdatasetVisualization.png",dpi=300) 
        
if __name__ == '__main__':
    
        #loading data..............   
        train_label,test_label,time_step, num_dim, train_data,test_data=Dataset()         
        if np.mat(train_data).shape[1]!=2:
            print("Visualization of kernel only supports for 2-Dim synthetic dataset")
            exit(-1)    #('Visualization of kernel only supports for 2D data')
        else:    
            Visualization_kernel(KernelFunction.CalKernel_(np.mat(train_data)))
        print(".....Visualizaiton End.....\n")

        
