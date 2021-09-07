#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:13:58 2021

@author: sunhaoxian
"""
#import needed packages
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pandas as pd

# A class used to generate display the original data set for further use
class Data:
    #INITIALIZE
    def __init__(self, name):
        self.name = name
        self.out = {}
    
    #read in file
    def write_json(self):
        with open(self.name, "w") as write_file:
            json.dump(self.out, write_file)
            
    #generate the age, weight and is_adult data
    def partition(self):
        self.out["xlabel"]="age"
        self.out["ylabel"]="weight"
        N=250; xmin=3; xmax=100; SF=0.12
        self.x = np.linspace(xmin, xmax, N)
        y = 181.0/(1+np.exp(-(self.x-13)/4))+20
        noise = SF*(max(y)-min(y))*np.random.uniform(-1,1,size=len(self.x))
        self.yn = y + noise
        self.out["x"] = self.x.tolist()
        self.out["y"] = self.yn.tolist()
        A_or_C = [] 
        for i in range(0,len(self.x)):
            if(self.x[i]<18):
                A_or_C.append(0)
            else:
                A_or_C.append(1)
        self.out["is_adult"] = A_or_C
        
    #display the original dataset
    def plot(self):
        fig, ax = plt.subplots()
        #Note: x here is weight and y is whether the person is child or adult
        ax.plot(self.yn, self.out["is_adult"], 'o', label = "weight")
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel(self.out["ylabel"], fontsize=FS)
        plt.ylabel("Adult = 1 Child = 0", fontsize=14)
        plt.title("Original Dataset")
        plt.show()
        
        
#2nd Logistic regression to predict a person is child or adult according his weight
class Logistic2:
    w_data = Data("weight.json")
    w_data.write_json()
    w_data.partition()
    iterations=[]
    loss_train=[]
    loss_val=[]
    ite = 0
    
    #initialize the class by setting x and y
    def __init__(self):
        self.x = self.w_data.yn
        self.y = self.w_data.out["is_adult"]

    #split the dataset into 2 parts
    def split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20)

    #sigmoid funciton used to generate the predicted y
    def modelFunc(self, x, p):
        return [p[0]+p[1]*(1.0/(1.0+np.exp(-(i-p[2])/(p[3]+0.00001)))) for i in x]
    
    #The loss function to record and return the mean squard error between the true values 
    #and predicted values. Further, by applying minimize to this function, we can find
    #the best p that can minize the mean square error and store it as our linear 
    #regression function. 
    def loss(self, p):
        y_pred = self.modelFunc(self.x_train, p)
        train_loss = mean_squared_error(self.y_train, y_pred)
        self.loss_train.append(train_loss)
        y_test_pred = self.modelFunc(self.x_test, p)
        test_loss = mean_squared_error(self.y_test, y_test_pred)
        self.loss_val.append(test_loss)
        self.ite += 1
        self.iterations.append(self.ite)
        return train_loss

    #display the loss change as the number of iterations increases
    def displayLoss(self):
        fig, ax = plt.subplots()
        ax.plot(self.iterations, self.loss_train, '--', label='train loss')
        ax.plot(self.iterations, self.loss_val, '-', label='test loss')
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('Number of Iterations', fontsize=FS)
        plt.ylabel('Loss', fontsize=FS)
        plt.title("Loss vs Iterations")
        plt.show()
        
    
#Main function
if __name__ == "__main__":
    data = Data("weight.json")
    data.write_json()
    data.partition()
    data.plot()
    
    #initialize the logistic regression
    log = Logistic2()
    log.split()

    #An initial point where it is easier to get to the optimal point
    po=[0.0, 1.0, 160.0, 10.0]
    #obtain the best coefficients
    res = minimize(log.loss, po, method='BFGS', tol=1e-15)
    popt=res.x
    print("OPTIMAL PARAM:",popt)
    #display the loss change
    log.displayLoss()
    
    #generate the logistic regression based on whole dataset
    x = log.x
    y_pred = log.modelFunc(x, popt)
    #sort the dataset to clearly display the plot
    df = pd.DataFrame(list(zip(x, y_pred)),
        columns =['weight', 'is_adult'])
    df.sort_values(by=['weight'], inplace = True)
    
    #display the final result
    fig, ax = plt.subplots()
    ax.plot(log.x_train, log.y_train, 'g*', label = "train set")
    ax.plot(log.x_test, log.y_test, 'rx', label = "test set")
    ax.plot(df['weight'], df['is_adult'], 'k', label='logistic regression')
    ax.legend()
    FS=18   #FONT SIZE
    plt.xlabel("weight", fontsize=FS)
    plt.ylabel("Adult = 1 Child = 0", fontsize=14)
    plt.title("Logistic Regression")
    plt.show()
    
