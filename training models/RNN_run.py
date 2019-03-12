
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

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.tools as tls
import plotly.io as pio
import plotly.graph_objs as go
from plotly.graph_objs import *
#from IPython.display import HTML

## DL Framework
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
##### Predefined models
import model_HPS1 as model 
import TrVaTe as TVT 


###GPU enabling and device allocation
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(1)


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

def plot_DLauc_perf(train_auc_allepv,test_auc_allepv,valid_auc_allepv,title_m):
    epochs=100
    train_auc_fg= go.Scatter(x= np.arange(epochs), y=train_auc_allepv, name='train')
    test_auc_fg= go.Scatter(x= np.arange(epochs), y=test_auc_allepv, name='test')
    valid_auc_fg= go.Scatter(x= np.arange(epochs), y=valid_auc_allepv, name='valid')
    valid_max = max(valid_auc_allepv)
    test_max = max(test_auc_allepv)
    data = [train_auc_fg,test_auc_fg,valid_auc_fg]
    layout = go.Layout(xaxis=dict(dtick=1),title=title_m)
    layout.update(dict(annotations=[go.layout.Annotation(text="Max Valid", x=valid_auc_allepv.index(valid_max), y=valid_max)]))
    fig = go.Figure(data=data, layout=layout)
    #iplot(fig, filename=title_m)
    pio.write_image (fig,title_m)

### for tracking computational timing
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


### DL RNN Model training 

def run_dl_model(ehr_model,train_sl,valid_sl,test_sl,bmodel_pth,bmodel_st):

    ##Hyperparameters -- Fixed for testing purpose
    epochs = 100
    l2 = 0.0001
    lr = 0.01
    eps = 1e-4
    w_model='RNN'
    optimizer = optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)   

    ##Training epochs
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep =[]
    valid_auc_allep =[]
    test_auc_allep=[]  
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = TVT.train(train_sl, model= ehr_model, optimizer = optimizer, batch_size =128)
        avg_loss = np.mean(train_loss)
        train_time = timeSince(start)
        eval_start = time.time()
        Train_auc, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = train_sl, which_model = w_model, batch_size = 128)
        valid_auc, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = valid_sl, which_model = w_model, batch_size = 128)
        TestAuc, y_real, y_hat = TVT.calculate_auc(model = ehr_model, data = test_sl, which_model = w_model, batch_size = 128)
        eval_time = timeSince(eval_start)
        print ("Epoch: " ,str(ep) ," Train_auc :" , str(Train_auc) , " , Valid_auc : " ,str(valid_auc) , " ,& Test_auc : " , str(TestAuc) ," Avg Loss: " ,str(avg_loss), ' , Train Time :' , str(train_time) ,' ,Eval Time :' ,str(eval_time))
        train_auc_allep.append(Train_auc)
        valid_auc_allep.append(valid_auc)
        test_auc_allep.append(TestAuc)

        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAuc
            ###uncomment the below lines to save the best model parameters
            best_model = ehr_model
            torch.save(best_model, bmodel_pth)
            torch.save(best_model.state_dict(), bmodel_st)
        if ep - bestValidEpoch >5: break
    print( 'bestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch))
    return train_auc_allep,valid_auc_allep,test_auc_allep



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


  ehr_model = model.EHR_RNN(input_size_1, embed_dim=128, hidden_size=128, n_layers=1, dropout_r=0., cell_type='GRU', bii=True , time=True)
  if use_cuda: ehr_model = ehr_model.cuda()    

  ### Model training 
  train_auc_allep,valid_auc_allep,test_auc_allep=run_dl_model(ehr_model,train_sl,valid_sl,test_sl,outFile+'.pth',outFile+'.st')
  plot_DLauc_perf(train_auc_allep,test_auc_allep,valid_auc_allep,outFile+'.png')