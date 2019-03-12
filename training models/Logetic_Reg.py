
### Tools and Packages
##Basics
import pandas as pd
import numpy as np
import sys, random
import math
from optparse import OptionParser
try:
    import cPickle as pickle
except:
    import pickle
import string
import re
import os
import time

## ML and Stats 
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
import sklearn.linear_model  as lm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import export_graphviz

## Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm




###GPU enabling and device allocation diabled for LR
use_cuda = False

# In[11]:


### Model Evaluation
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    pred_prob=model.predict_proba(test_features)
    auc_p=roc_auc_score(test_labels,pred_prob[:,1])
    print('Model Performance')
    print('AUC = {:0.2f}%.'.format(auc_p*100))
    return test_labels,pred_prob[:,1]


# In[12]:


### ROC curve plotting

def plot_roc_curve(label,score,filename):
    fpr, tpr, ths = m.roc_curve(label, score) ### If I round it gives me an AUC of 64%
    roc_auc = m.auc(fpr, tpr)
    print('AUC',roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(filename,dpi=199)



### for tracking computational timing
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    
  filePrefix= sys.argv[1]
  typeFile= sys.argv[2]
  outFile = sys.argv[3]
  parser = OptionParser()
  (options, args) = parser.parse_args()
  
  
  ### A sample one hospital data, this is preprocessed data 
  
  train_sl= pickle.load(open(filePrefix+'.train', 'rb'), encoding='bytes')
  test_sl= pickle.load(open(filePrefix+'.test', 'rb'), encoding='bytes')
  valid_sl= pickle.load(open(filePrefix+'.valid', 'rb'), encoding='bytes')
  print (len(train_sl),len(valid_sl),len(test_sl))
  
  # types dictionary
  types_d=pickle.load(open(typeFile, 'rb'), encoding='bytes')
  types_d_rev = dict(zip(types_d.values(),types_d.keys()))
  input_size_1=[len(types_d_rev)+1]
  print(input_size_1)
  
  
  # In[10]:
  
  
  ##### Data conversion to onehot matrices for Logestic Regression and may be Random Forest Basic test
  pts_tr=[]
  labels_tr=[]
  features_tr=[]
  for pt in train_sl:
      pts_tr.append(pt[0])
      labels_tr.append(pt[1])
      x=[]
      for v in pt[-1]:
          x.extend(v[-1])
      features_tr.append(x)
        
  pts_t=[]
  labels_t=[]
  features_t=[]
  for pt in test_sl:
      pts_t.append(pt[0])
      labels_t.append(pt[1])
      x=[]
      for v in pt[-1]:
          x.extend(v[-1])
      features_t.append(x)
  
      
  mlb = MultiLabelBinarizer(classes=range(input_size_1[0])[1:])
  nfeatures_tr = mlb.fit_transform(features_tr)
  nfeatures_t= mlb.fit_transform(features_t)
  

  
  
  # ### Logestic Regression
  
  # In[18]:
  EHR_LR= lm.LogisticRegression()
  start = time.time()
  EHR_LR.fit(nfeatures_tr, labels_tr)
  train_time = timeSince(start)
  eval_start=time.time()
  labels,scores=evaluate(EHR_LR,nfeatures_t, labels_t)
  eval_time = timeSince(eval_start)
  
  plot_roc_curve(labels,scores,outFile)
  
  print ('train_time:', train_time , ', eval_time:',eval_time)
  print ("Highly contributing factors based on coefficients: ")
  f_imp=EHR_LR.coef_
  ## top contributing factors in both directions
  for i,j in enumerate(f_imp[0].tolist()):
      if j>1.5 or j<-1.5:
          print (i+1,j,types_d_rev[i+1])
  
  
