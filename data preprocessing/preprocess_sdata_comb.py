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
    
   caseFile1= sys.argv[1]
   controlFile1= sys.argv[2]
   caseFile2= sys.argv[3]
   controlFile2= sys.argv[4]
   typeFile= sys.argv[5]
   outFile = sys.argv[6]
   subsetType = sys.argv[7]
   ptlfile = sys.argv[8]
   parser = OptionParser()
   (options, args) = parser.parse_args()
   
   #_start = timeit.timeit()
   
   debug=False
   np.random.seed(1)
   #visit_list = []
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []

   
   ### Building the Data starting from the matching file

   print (" Loading cases and controls" )
   
   ## loading Case
  
  # caseFile="samp_hf50_case.csv"
   print ('Loading cases')
   data_case1 = pd.read_table(caseFile1)
   data_case2 = pd.read_table(caseFile2)
   data_case1.columns = ["Pt_id", "ICD", "Time"]
   data_case2.columns = ["Pt_id", "ICD", "Time"]
   data_case= pd.concat([data_case1,data_case2])
   data_case.columns = ["Pt_id", "ICD", "Time"]
   data_case['Label'] = 1
   set_p = pickle.load(open(ptlfile, 'rb'))#,encoding= 'bytes')
   ncases_sub=data_case[data_case["Pt_id"].isin(set_p)]


    ## loading Control
#   controlFile = "samp_hf50_control.csv"    
   print ('Loading controls')
   data_control1 = pd.read_table(controlFile1)
   data_control2 = pd.read_table(controlFile1)
   data_control1.columns = ["Pt_id", "ICD", "Time"]
   data_control2.columns = ["Pt_id", "ICD", "Time"]
   data_control= pd.concat([data_control1,data_control2])
   data_control.columns = ["Pt_id", "ICD", "Time"]
   data_control['Label'] = 0
   nctrls_sub=data_control[data_control["Pt_id"].isin(set_p)]


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
            #print (group)
            data_dt_c = []
            #for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False): ### changing the sort order
            for Time, subgroup in group.sort_values(['Time'], ascending=True).groupby('Time', sort=False): ### changing the sort order
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))

            if len(data_i_c) > 0:
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
#                 print 'Converting cerner codes to a sequential integer code, and creating the types dictionary'
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
                  #cs.append(newPatient_c)

 #                 print cs
                                                            
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

   ptee_list = []
   
  
    ### Creating the full pickled lists

   pickle.dump(label_list, open(outFile+'.labels.'+subsetType, 'wb'), -1)
   pickle.dump(newVisit_list, open(outFile+'.visits.'+subsetType, 'wb'), -1)
   pickle.dump(types, open(outFile+'.types.'+subsetType, 'wb'), -1)
   pickle.dump(pt_list, open(outFile+'.pts.'+subsetType, 'wb'), -1)
   pickle.dump(dur_list, open(outFile+'.days.'+subsetType, 'wb'), -1)

   ### Create the combined list for the Pytorch RNN
   print ('Reparsing')
   fset=[]
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
   
   combined_file_name=outFile+'.combined.'+subsetType
   pickle.dump(fset, open(combined_file_name, 'wb'), -1)

