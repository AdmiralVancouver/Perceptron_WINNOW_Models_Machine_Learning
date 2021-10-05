
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
    print("Peppy's Final Weights: ",self.weights)

  def predict(self, feature):
     linear_output =np.dot(feature, self.weights) + self.bias
     class_predicted = self.activationFn(linear_output)
     return class_predicted

  def stepFn(self, features):
    return np.where(features >=0, 1, 0)





#END OF CLASS
def accuracy(y_true, y_predicted):
  accuracy = np.sum(y_true == y_predicted)/ len(y_true)
  return accuracy

#Data comes from csv so lets get that good good dataFrame
mushroomDFHeaders = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
                      'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
                      'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number',
                      'ring-type','spore-print-color','population','habitat']
  
mushroomDataFrame = pd.read_csv('noisy_mushrooms.csv',names=mushroomDFHeaders)

#Copying dataframe to maintain DF integrity
mushroomDFCopy = mushroomDataFrame.copy()
#Let's see what's the shape of our nice little Dataframe
#Will be in format of '(rows,columns)'
print(mushroomDataFrame.shape)




#Now time to normalize the data. From discrete to numeric, we need encoders
encodingFeatures = preprocessing.OrdinalEncoder()
encodingClasses = preprocessing.LabelEncoder()


print(encodingClasses)
print(encodingFeatures)



#Time to extract and tranform that data. Separating attributes from class
#variable names self descriptive
x_attributes = mushroomDFCopy.iloc[1:, 1:].values
x_enc_attributes = encodingFeatures.fit(x_attributes)
  #Encoding DEBUG
print(x_attributes)
x_enc_attributes = encodingFeatures.transform(x_attributes)
#Transform DEBUG
print(x_enc_attributes)


y_class = mushroomDFCopy.iloc[1:,0].values
print("Y-Class: ",y_class)
y_enc_class = encodingClasses.fit(y_class)
#Encoding DEBUG
y_enc_class = encodingClasses.transform(y_class)
print(y_enc_class)



#Training/Testing time!
#Let's split the data into train vs test
#To Stratify is, quote:
#each set contains approximately the same percentage 
#of samples of each target class as the complete set.
#So let's do that
x_train,x_test,y_train,y_test = train_test_split(x_enc_attributes,y_enc_class, test_size=0.20, stratify=y_enc_class)

#Creating instance of perceptron. I'll name it Peppy
peppy = Perceptron(lr=0.02, epochs=1000)

  #Fit and display our accuracy score! Hope Peppy did well!
peppy.fit(x_train, y_train)
predictionTrain = peppy.predict(x_test)
print("Peppy's Accuracy: ", accuracy(y_test, predictionTrain))

peppyPlot = plt.figure()
plt.plot(x_train[:,0] [y_train==0], x_train[:, 1] [y_train == 0], 'r^')
plt.plot(x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], 'bs')
plt.title('Perceptron Classification')

plt.savefig("BringDaNoise_Perceptron_Learning_Plot.png")









  #DEBUG
  #print(x_attributes.head(10))
  #print(y_class.head(10))

