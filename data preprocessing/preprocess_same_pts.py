'''
# LRasmy @Zhilab 3/12/2019
#
# This script processes originally extracted data for example from Cerner HealthFacts Dataset
# and builds pickled lists including a full list that includes all information for case and controls
# Similar to the original preprocessing code available under https://github.com/ZhiGroup/pytorch_ehr/blob/master/Preprocessing/data_preprocessing_v1.py
# it outputs pickled list of the following shape,
#[[pt1_id,label,[
#                  [[delta_time 0],[list of Medical codes in Visit0]],
#                  [[delta_time between V0 and V1],[list of Medical codes in Visit2]],
#                   ......]],
# [pt2_id,label,[[[delta_time 0],[list of Medical codes in Visit0 ]],[[delta_time between V0 and V1],[list of Medical codes in Visit2]],......]]]
#
# The main difference of this script, is it consider limiting the output to a predefined set of patients which we need for comparison purposes
# Therefore, we feed this script with Case file and Control files each is just a three columns like pt_id | medical_code | visit_date along with the predefined set of patients and execute like:
#
# python preprocess_same_pts.py <Case File> <Control File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <subset_type{'train','valid','test'}> <patient_subset_list>
#
'''



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
   subsetType = sys.argv[5]
   ptlfile = sys.argv[6]
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
  
   print ('Loading cases')
   data_case = pd.read_table(caseFile)
   data_case.columns = ["Pt_id", "ICD", "Time"]
   data_case['Label'] = 1
   set_p = pickle.load(open(ptlfile, 'rb'))#,encoding= 'bytes')
   ncases_sub=data_case[data_case["Pt_id"].isin(set_p)]


    ## loading Control
   print ('Loading controls')
   data_control = pd.read_table(controlFile)
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
  
   #print ("consumed time for data loading",(_start -end_time)/1000.0 )
    
   full_list=[]
   index_date = {}
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []
   dur_list=[]
   newVisit_list = []
   count=0
   
   for Pt, group in data_l.groupby('Pt_id'):
            data_i_c = []
            data_dt_c = []
            for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False): ### ascending=True normal order ascending=False reveresed order
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
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
                             #xx = ((dt.strptime(data_dt_c[jx-1], '%d-%b-%y'))-(dt.strptime(data_dt_c[jx], '%d-%b-%y'))).days ## use if original data have time information or different date format
                             #xx = (data_dt_c[jx]- data_dt_c[jx-1]).days ### normal order
                             xx = (data_dt_c[jx-1] - data_dt_c[jx]).days ## reversed order                            
                             v_dur_c.append(xx)
            
            ### Diagnosis recoding
            newPatient_c = []
            for visit in data_i_c:
                      newVisit_c = []
                      for code in visit:
                    				if code in types: newVisit_c.append(types[code])
                    				else:                             
                    					  types[code] = len(types)+1
                    					  newVisit_c.append(types[code])
                      newPatient_c.append(newVisit_c)
                                                            
            if len(data_i_c) > 0: ## only save non-empty entries
                  label_list.append(group.iloc[0]['Label'])
                  pt_list.append(Pt)
                  newVisit_list.append(newPatient_c)
                  dur_list.append(v_dur_c)
 
            count=count+1
            if count % 1000 == 0: print ('processed %d pts' % count)

   
    ### Creating the full pickled lists ### uncomment if you need to dump the all data before splitting

   pickle.dump(types, open(outFile+'.types.'+subsetType, 'wb'), -1)
        
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
   cfilename=outFile+'.combined.'+subsetType
   pickle.dump(fset, open(cfilename, 'wb'), -1)
