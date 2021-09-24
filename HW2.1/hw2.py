#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:26:59 2021

@author: sunhaoxian
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import random

#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"


OPT_ALGO='BFGS'	#HYPER-PARAM

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
NFIT=4; X_KEYS=['x']; Y_KEYS=['y']


#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
p=np.random.uniform(0.5,1.,size=NFIT)

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#------------------------
#DATA CLASS to readin, normalize and partition the dataset
#------------------------
class Data:
    #read in the csv file
	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			with open(FILE_NAME, errors='ignore') as f:
				self.input = json.load(f)  #read into dictionary

			#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
			X=[]; Y=[]
			for key in self.input.keys():
				if(key in X_KEYS): X.append(self.input[key])
				if(key in Y_KEYS): Y.append(self.input[key])

			#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
			self.X=np.transpose(np.array(X))
			self.Y=np.transpose(np.array(Y))
			self.been_partitioned=False

			#INITIALIZE FOR LATER
			self.YPRED_T=1; self.YPRED_V=1

			#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
			self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
			self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 
		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED"); 

    #divide the dataset into 2 parts
	def partition(self,f_train=0.8, f_val=0.2):
		if(f_train + f_val != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0])
		CUT1=int(f_train*self.X.shape[0]); 
		train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
		self.xt=self.X[train_idx]; self.yt=self.Y[train_idx]; self.xv=self.X[val_idx];   self.yv=self.Y[val_idx]
    
    #normalize the dataset
	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD 
		self.Y=(self.Y-self.YMEAN)/self.YSTD  



def unnormalize(x, mean, std):
    #unnormalize the normalized data
    return std * x + mean 
        
    

def model(x, p):
    #The base formula of logistic regression
    return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))
		
		      
         
def loss(p, xb, yb):
    #The loss function of MSE
    yp = model(xb, p)
    training_loss = (np.mean((yp-yb)**2.0))
    return training_loss


iteration=0; iterations=[]; loss_train=[];  loss_val=[]
def optimizer(objective, xt, yt, xv, yv, algo = 'GD', LR = 0.001, method = 'batch'):
    #Function to obtain the coefficients and record the loss
    global iterations,loss_train,loss_val,iteration
    p0=np.random.uniform(0.1,1.,size=NFIT)
    dx=0.001	
    LR=0.01
    t=0
    tmax=30000
    tol=10**-10
    xi=p0
    previous_step=0
    NDIM=len(p0)
    alpha = 0.3
        
    print("INITAL GUESS: ",xi)
        
    while(t <= tmax):
        #use the whole set as training set
        if(method == 'batch' and t == 0):
            xb1 = xt; yb1 = yt
            
        #use each row as a training data in each iteration
        if(method == 'stocastic'):
            if(t == 0):
                index_to_use = 0
            else:
                if(index_to_use == len(xt)-1):
                    index_to_use = 0
                else:
                    index_to_use += 1
            xb1 = xt[index_to_use]; yb1 = yt[index_to_use]
            
        #use half of the dataset to train the model
        if(method == 'minibatch'):
            #randomly choose 1/2 data to train the model
            tmp_idx = random.choices(range(len(xt)), k = (int)(len(xt)/2))
            xb1 = xt[tmp_idx]
            yb1 = yt[tmp_idx]
                
        #obtain the value of gradient vector
        df_dx=np.zeros(NDIM)
        for i in range(0,NDIM):
            dX=np.zeros(NDIM);
            dX[i]=dx; 
            xm1=xi-dX; 
            df_dx[i]=(objective(xi,xb1,yb1) - objective(xm1,xb1,yb1))/dx
            
        #deal with Gradient Descent algorithm
        if(algo == 'GD'):
            xip1=xi - LR * df_dx #STEP 
            
        #deal with Gradient Descent Momentum Method
        if (algo == 'GD-Mom'):
            previous_step = LR * df_dx + alpha * previous_step
            xip1 = xi - previous_step

        #store the loss into corresponding arrays
        if(t%2==0):
            df=np.mean(np.absolute(objective(xip1,xb1,yb1)-objective(xi,xb1,yb1)))
            yp=model(xt,xi) #model predictions for given parameterization p
            training_loss=(np.mean((yp-yt)**2.0))  #MSE
            yp=model(xv,xi) #model predictions for given parameterization p
            validation_loss=(np.mean((yp-yv)**2.0))  #MSE
            loss_train.append(training_loss); loss_val.append(validation_loss)
            iterations.append(iteration); iteration+=1
            
        #if the coefficients is already good enough
        if(df<tol):
            print("STOPPING CRITERION MET (STOPPING TRAINING)")
            break
        
        #update the coefficients
        xi=xip1
        t=t+1
        
    return xi
                
                

                
#------------------------
#MAIN 
#------------------------
D = Data(INPUT_FILE)		#INITIALIZE DATA OBJECT 



D.normalize()				#NORMALIZE
D.partition()
coefs = optimizer(loss, D.xt, D.yt, D.xv, D.yv, algo='GD', LR = 0.001, method='minibatch')
print("Final Guess: ", coefs)
xm = np.array(sorted(D.xt))
yp = np.array(model(xm,coefs))


#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnormalize(D.xt, D.XMEAN, D.XSTD), unnormalize(D.yt, D.YMEAN, D.YSTD), 'o', label='Training set')
	ax.plot(unnormalize(D.xv, D.XMEAN, D.XSTD), unnormalize(D.yv, D.YMEAN, D.YSTD), 'x', label='Validation set')
	ax.plot(unnormalize(xm, D.XMEAN, D.XSTD),unnormalize(yp, D.YMEAN, D.YSTD), '-r', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(D.xt,coefs), D.yt, 'o', label='Training set')
	ax.plot(model(D.xv,coefs), D.yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()
    
exit()
