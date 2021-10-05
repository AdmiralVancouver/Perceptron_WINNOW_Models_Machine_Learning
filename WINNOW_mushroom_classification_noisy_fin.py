#!/usr/bin/env python
# coding: utf-8

# In[64]:


# %load WINNOW_mushroom_classification.py

#### %load perceptron_mushroom_classification.py
from __future__ import print_function
from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt #Shows the image
import sys
import numpy as np
import pandas as pd
import pickle #Read in and out
import random

#Bad practice yes but for the sake of simplicity, we'll have the perceptron class
#in with the main
class Perceptron:

  def __init__(self, lr=0.01, epochs=10):
    #Learning Rate
    self.lr = lr
    #Epochs/Iterations
    self.epochs = epochs
    #Activation Function
    self.activationFn = self.stepFn
    self.weights = None
    self.bias = None

  def fit(self,features,classifiers):
    #Set variables 
    numSamples, numFeatures = features.shape

    #Initiate some of those parameters
    self.weights = np.zeros(numFeatures)
    self.bias = 0

    classifiers_ = np.array([1 if i > 0 else 0 for i in classifiers])
    
    #the underscore ( _ ) is just a filler variable since we don't use it
    #anywhere else
    for _ in range(self.epochs):

      for idx, x_i in enumerate(features):

        linear_output = np.dot(x_i, self.weights) + self.bias
        class_predicted = self.activationFn(linear_output)

        #Update rules
        update = self.lr * (classifiers_[idx] - class_predicted)

        self.weights += update * x_i
        self.bias += update


  def predict(self, feature):
     linear_output =np.dot(feature, self.weights) + self.bias
     class_predicted = self.activationFn(linear_output)
     return class_predicted

  def stepFn(self, features):
    return np.where(features >=0, 1, 0)

#END OF CLASS

#Bad practice continues but this time in WINNOW!
class Winnow:

  def __init__(self, lr=0.02, epochs=10):
    #Learning Rate
    self.lr = lr
    #Epochs/Iterations
    self.epochs = epochs
    #Activation Function
    self.activationFn = self.stepFn
    self.weights = None
    self.threshold = None

#The only method that changes - Away goes the addition, enter Multiplication
  def fit(self,features,classifiers):
    #Set variables 
    numSamples, numFeatures = features.shape # lows and columns 

    #Initiate some of those parameters
    self.weights = np.ones(numFeatures) # initiate all weights as 1
    self.threshold = numFeatures - 0.1 # set threshold Theta 

    classifiers_ = np.array([1 if i > 0 else 0 for i in classifiers])
    
    #the underscore ( _ ) is just a filler variable since we don't use it
    #anywhere else
    for _ in range(self.epochs):

      for idx, x_i in enumerate(features):
        linear_output = np.dot(x_i, self.weights) - self.threshold
        class_predicted = self.activationFn(linear_output)

        #Update rules with alpha = 2
        
        if classifiers_[idx] > class_predicted:
          update = 2
        elif classifiers_[idx] < class_predicted:
          update = 0.5
        else:
          update = 1

        
        #print(self.weights)
        for ii in range(len(x_i)): # modify weights only if the feature == 1
            if x_i[ii] == 1:
             self.weights[ii] = self.weights[ii] * update
            

  def predict(self, feature):
     linear_output =np.dot(feature, self.weights) - self.threshold 
     class_predicted = self.activationFn(linear_output)
     return class_predicted

  def stepFn(self, features):
    return np.where(features >0, 1, 0) # 1, when the linear_output > threshold 


#END OF CLASS
def accuracy(y_true, y_predicted):
  accuracy = np.sum(y_true == y_predicted)/ len(y_true)
  return accuracy


# In[65]:


#Data comes from csv so lets get that good good dataFrame
mushroomDFHeaders = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
                      'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
                      'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number',
                      'ring-type','spore-print-color','population','habitat']
  
mushroomDataFrame = pd.read_csv('mushrooms.csv',names=mushroomDFHeaders)
#mushroomDataFrame = pd.read_csv('noisy_mushrooms.csv',names=mushroomDFHeaders) # for noisy stuff

#Copying dataframe to maintain DF integrity
mushroomDFCopy = mushroomDataFrame.copy()
#Let's see what's the shape of our nice little Dataframe
#Will be in format of '(rows,columns)'
print(mushroomDataFrame.shape)


# In[66]:


#Now time to normalize the data. From discrete to numeric, we need encoders
p_encodingFeatures = preprocessing.OrdinalEncoder()
w_encodingFeatures = preprocessing.OneHotEncoder(handle_unknown='ignore')
encodingClasses = preprocessing.LabelEncoder()

#print(encodingClasses)
#print(encodingFeatures)


# In[67]:


#Time to extract and tranform that data. Separating attributes from class for PERCEPTRON
#variable names self descriptive
x_attributes = mushroomDFCopy.iloc[1:, 1:].values
p_x_enc_attributes = p_encodingFeatures.fit(x_attributes)
  #Encoding DEBUG
print(x_attributes)
p_x_enc_attributes = p_encodingFeatures.transform(x_attributes)
#Transform DEBUG
print(p_x_enc_attributes)


# In[68]:


#Time to extract and tranform that data. Separating attributes from class for WINNOW
#variable names self descriptive
w_x_enc_attributes = w_encodingFeatures.fit(x_attributes)
#print(encodingFeatures.categories_)
  #Encoding DEBUG
print(x_attributes)
w_x_enc_attributes = w_encodingFeatures.transform(x_attributes).toarray()
#Transform DEBUG
print(w_x_enc_attributes)


# In[69]:


y_class = mushroomDFCopy.iloc[1:,0].values
print("Y-Class: ",y_class)
y_enc_class = encodingClasses.fit(y_class)
#Encoding DEBUG
y_enc_class = encodingClasses.transform(y_class)
print(y_enc_class)


# In[70]:


#Adding noise to the entire set?.?
#Adding errors only in the training set
noise = 0.10 # n% of ratio
noise_sample = round(y_enc_class.shape[0]*noise) # n% of the training set
idx_ = np.arange(0,y_enc_class.shape[0]-1).tolist() # for indexing noise
noise_idx = random.sample(idx_,noise_sample) 
# let's make some noise
for __ in noise_idx:
    y_enc_class[__] = 1 - y_enc_class[__]


# In[71]:


#Training/Testing time!
#Let's split the data into train vs test
#To Stratify is, quote:
#each set contains approximately the same percentage 
#of samples of each target class as the complete set.
#So let's do that
p_x_train,p_x_test,p_y_train,p_y_test = train_test_split(p_x_enc_attributes,y_enc_class, test_size=0.20, stratify=y_enc_class)
w_x_train,w_x_test,w_y_train,w_y_test = train_test_split(w_x_enc_attributes,y_enc_class, test_size=0.20, stratify=y_enc_class)


# In[52]:


#Creating instance of perceptron. I'll name it Peppy
peppy = Perceptron(lr=0.02, epochs=500)

  #Fit and display our accuracy score! Hope Peppy did well!
peppy.fit(p_x_train, p_y_train)
p_predictionTrain = peppy.predict(p_x_test)
print(peppy.weights)
print("Peppy's Accuracy: ", accuracy(p_y_test, p_predictionTrain))

peppyPlot = plt.figure()
plt.plot(p_x_train[:,0] [p_y_train==0], p_x_train[:, 1] [p_y_train == 0], 'r^')
plt.plot(p_x_train[:, 0][p_y_train == 1], p_x_train[:, 1][p_y_train == 1], 'bs')
plt.title('Perceptron Classification with 10% noise')

plt.savefig("BringDaNoise_Perceptron_Learning_Plot.png")


# In[72]:


#Creating instance of Winnow-using Perceptron. Enter Willow the Winnow!
willow = Winnow(lr=0.02, epochs=500)
#print(x_train)
  #Fit and display our accuracy score! Hope Peppy did well!
willow.fit(w_x_train, w_y_train)

w_predictionTrain = willow.predict(w_x_test)
print(willow.weights)
print("Willow's Accuracy: ", accuracy(w_y_test, w_predictionTrain))

willowPlot = plt.figure()
plt.plot(w_x_train[:,0] [w_y_train==0], w_x_train[:, 1] [w_y_train == 0], 'r^')
plt.plot(w_x_train[:, 0][w_y_train == 1], w_x_train[:, 1][w_y_train == 1], 'bs')
plt.title('Winnow Classification with 10% noise')

plt.savefig("WINNOW_Learning_Plot.png")

  #DEBUG
  #print(x_attributes.head(10))
  #print(y_class.head(10))


# In[ ]:




