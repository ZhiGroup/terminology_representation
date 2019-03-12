# This script processes Cerner dataset and builds pickled lists including a full list that includes all information for case and controls
# The output data are cPickled, and suitable for training Doctor AI or RETAIN
# Using similar logic of process_mimic Written by Edward Choi (mp2893@gatech.edu) ### updated by LRasmy
# Usage: feed this script with Case file, Control file ,and Case-Control Matching file. and execute like:
# python process_cerner_f_5.py <Case File> <Control File> <Matching File > <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <train/test/valid>
# Case and contol files should contain pt_id, Diagnosis, Date
# the matching file should contain the case_id, assigned control_id , index_date

# Output files
# <output file>.pts: List of unique Cerner Patient IDs. Created for validation and comparison purposes
# <output file>.labels: List of binary values indicating the label of each patient (either case(1) or control(0)) #LR
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.days: List of List of integers representing number of days between consequitive vists. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.visits: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# The above files will also be splitted to train,validation and Test subsets using the Ratio of 75:10:15
# For further resampling and Sorting use the sampling_cerner.py


import sys
from optparse import OptionParser

try:
    import cPickle as pickle
except:
    import pickle

#import pprint

import numpy as np
import random
import pandas as pd
#from pandas import read_table
#from pandas import dataframe
from datetime import datetime as dt
from datetime import timedelta
#import gzip

#import timeit
if __name__ == '__main__':
    
   caseFile= sys.argv[1]
   controlFile= sys.argv[2]
   typeFile= sys.argv[3]
   outFile = sys.argv[4]
   samplesize_case = int(sys.argv[5])
   samplesize_ctrl = int(sys.argv[6])
   parser = OptionParser()
   (options, args) = parser.parse_args()
 
   
   #_start = timeit.timeit()
   
   debug=False
   #np.random.seed(1)
   #visit_list = []
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []

   
   ### Building the Data starting from the matching file

   print (" Loading cases and controls" )
   

   
   ## loading Case
   print('loading cases')
  # caseFile="samp_hf50_case.csv"
   data_case = pd.read_table(caseFile)
   data_case.columns = ["Pt_id", "ICD", "Time"]
   data_case['Label'] = 1
   ##print(data_case[0:10])
   ##### cleaning up patients with 2> visits >100 and 3> codes>150 
   cs_f=data_case.groupby("Pt_id", as_index=False).agg({"ICD": pd.Series.nunique, "Time": pd.Series.nunique})
   ##print(cs_f[0:10])
   cs_ff=cs_f[(cs_f["Time"]>1) & (cs_f["Time"]<=100) & (cs_f["ICD"]>2) & (cs_f["ICD"]<=200)]
   ##print(cs_ff[0:10], type (cs_ff), cs_ff.columns)
   cas_sk=cs_ff["Pt_id"]
   ##print(cas_sk[0:10])
   cas_sk=cas_sk.drop_duplicates()
   print ('Case sampling')
   cas_sk_samp=cas_sk.sample(n=samplesize_case)
   ncases_sub=data_case[data_case["Pt_id"].isin(cas_sk_samp.values.tolist())]
   #print (len(ncases_sub),'sample ',  ncases_sub[0:10])


    ## loading Control
#   controlFile = "samp_hf50_control.csv"    
   print('loading ctrls')
   data_control = pd.read_table(controlFile)
   data_control.columns = ["Pt_id", "ICD", "Time"]
   data_control['Label'] = 0
   
   ### Exclude controls where the difference between the 2 last visits less than 30 days
   print('ctrls cleaning')   
   data_control=data_control.sort_values(["Pt_id", "Time"],ascending=False)
   dc_cl1=data_control[["Pt_id", "Time"]].drop_duplicates().groupby("Pt_id").head(2)
   dc_cl1["Time"]= pd.to_datetime(dc_cl1["Time"])
   dc_cl1['diff']=dc_cl1.groupby(["Pt_id"], as_index=False)["Time"].transform(lambda x: -x.diff())
   ##print(dc_cl1[0:10])
   new_ctl_pts_l = dc_cl1[dc_cl1['diff']>'30 days']["Pt_id"]
   new_data_control = data_control[data_control["Pt_id"].isin(new_ctl_pts_l.values.tolist())]
   ##print(new_data_control[0:10])
    
   ##### cleaning up patients with 2> visits >100 and 3> codes>150 
   ct_f=new_data_control.groupby("Pt_id", as_index=False).agg({"ICD": pd.Series.nunique, "Time": pd.Series.nunique})
   
   ### added 1 to visit counts  below as I will exclude the last encounter for controls later in line 190
   ct_ff=ct_f[(ct_f["Time"]>2) & (ct_f["Time"]<=101) & (ct_f["ICD"]>2) & (ct_f["ICD"]<=200)] 
   print('ctrls sampling')       
   ctr_sk=ct_ff["Pt_id"]
   ctr_sk=ctr_sk.drop_duplicates()
   ctr_sk_samp=ctr_sk.sample(n=samplesize_ctrl)
   nctrls_sub=data_control[data_control["Pt_id"].isin(ctr_sk_samp.values.tolist())]
   

#   data_l= pd.concat([data_case,data_control])   
   data_l= pd.concat([ncases_sub,nctrls_sub])
   
   ## loading the types
   
   if typeFile=='NA': 
       types={}
   else:
      with open(typeFile, 'rb') as t2:
             types=pickle.load(t2)
        
   #end_time = timeit.timeit()
    ## Mapping cases and controls
  
   #print ("consumed time",(_start -end_time)/1000.0 )
    
   full_list=[]
   index_date = {}
#  The_patient_of = {}
   #visit_list = []
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []
   dur_list=[]
   #types = {}
   newVisit_list = []
   count=0
   
   for Pt, group in data_l.groupby('Pt_id'):
            data_i_c = []
            data_dt_c = []
            for Time, subgroup in group.sort_values(['Time'], ascending=True).groupby('Time', sort=False): ### normal order
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
            #print ('dates', data_dt_c)
            if len(data_i_c) > 0:
                 # creating the duration in days between visits list, first visit marked with 0        
                  v_dur_c=[]
            if len(data_dt_c)<=1:
                     v_dur_c=[0]
            else:
                     for jx in range (len(data_dt_c)):
                         if jx==0:
                             v_dur_c.append(jx)
                         else:
                             #xx = ((dt.strptime(data_dt_c[jx-1], '%d-%b-%y'))-(dt.strptime(data_dt_c[jx], '%d-%b-%y'))).days
                             xx = (data_dt_c[jx]- data_dt_c[jx-1]).days                             
                             v_dur_c.append(xx)
            
            ### Diagnosis recoding
            #print ('dur', v_dur_c)                  
            newPatient_c = []
            for visit in data_i_c:
                      #print visit
                      newVisit_c = []
                      for code in visit:
                    				if code in types: newVisit_c.append(types[code])
                    				else:                             
                    					  types[code] = len(types)+1
                    					  newVisit_c.append(types[code])
                      #print types
                      #print newVisit_c 
                      newPatient_c.append(newVisit_c)
                                                            
            if len(data_i_c) > 0: ## only save non-empty entries
                  #visit_list.append(data_i_c)
                  label_list.append(group.iloc[0]['Label'])
                  pt_list.append(Pt)
                  #print(group.iloc[0]['Label'])
                  if group.iloc[0]['Label']==0: 
                      #print(newPatient_c,newPatient_c[:-1])
                      newVisit_list.append(newPatient_c[:-1])
                      dur_list.append(v_dur_c[:-1])
                  else:
                      newVisit_list.append(newPatient_c)
                      dur_list.append(v_dur_c)
 
            count=count+1
            if count % 1000 == 0: print ('processed %d pts' % count)

   
    ### Creating the full pickled lists

   #pickle.dump(label_list, open(outFile+'.labels.'+subsetType, 'wb'), -1)
   #pickle.dump(newVisit_list, open(outFile+'.visits.'+subsetType, 'wb'), -1)
   pickle.dump(types, open(outFile+'.types', 'wb'), -1)
   #pickle.dump(pt_list, open(outFile+'.pts.'+subsetType, 'wb'), -1)
   #pickle.dump(dur_list, open(outFile+'.days.'+subsetType, 'wb'), -1)

  
    ### Random split to train ,test and validation sets
   print ("Splitting")
   dataSize = len(pt_list)
   np.random.seed(0)
   ind = np.random.permutation(dataSize)
   nTest = int(0.2 * dataSize)
   nValid = int(0.1 * dataSize)
   test_indices = ind[:nTest]
   valid_indices = ind[nTest:nTest+nValid]
   train_indices = ind[nTest+nValid:]
    
   for subset in ['train','valid','test']:
       if subset =='train':
            indices = train_indices
       elif subset =='valid':
            indices = valid_indices
       elif subset =='test':
            indices = test_indices
       else: 
            print ('error')
            break
       subset_x = [newVisit_list[i] for i in indices]
       subset_y = [label_list[i] for i in indices]
       subset_t = [dur_list[i] for i in indices]
       subset_p = [pt_list[i] for i in indices]
       nseqfile = outFile +'.visits.'+subset
       nlabfile = outFile +'.labels.'+subset
       ntimefile = outFile +'.days.'+subset
       nptfile = outFile +'.pts.'+subset
       pickle.dump(subset_x, open(nseqfile, 'wb'), -1)
       pickle.dump(subset_y, open(nlabfile, 'wb'), -1)
       pickle.dump(subset_t, open(ntimefile, 'wb'), -1)
       pickle.dump(subset_p, open(nptfile, 'wb'), -1)    
        
    ### Create the combined list for the Pytorch RNN
   fset=[]
   print ('Reparsing')
   for pt_idx in range(len(pt_list)):
                pt_sk= pt_list[pt_idx]
                pt_lbl= label_list[pt_idx]
                pt_vis= newVisit_list[pt_idx]
                pt_td= dur_list[pt_idx]
                d_gr=[]
                n_seq=[]
                d_a_v=[]
                for v in range(len(pt_vis)):
                        nv=[]
                        nv.append([pt_td[v]])
                        nv.append(pt_vis[v])                   
                        n_seq.append(nv)
                n_pt= [pt_sk,pt_lbl,n_seq]
                fset.append(n_pt)              
    
   ### split the full combined set to the same as individual files

   train_set_full = [fset[i] for i in train_indices]
   test_set_full = [fset[i] for i in test_indices]
   valid_set_full = [fset[i] for i in valid_indices]
   ctrfilename=outFile+'.combined.train'
   ctstfilename=outFile+'.combined.test'
   cvalfilename=outFile+'.combined.valid'    
   pickle.dump(train_set_full, open(ctrfilename, 'wb'), -1)
   pickle.dump(test_set_full, open(ctstfilename, 'wb'), -1)
   pickle.dump(valid_set_full, open(cvalfilename, 'wb'), -1)
  

