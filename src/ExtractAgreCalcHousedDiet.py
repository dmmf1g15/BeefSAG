'''
Code to process AgreCalc data and pick out diet of animal types

'Enterprise Sector Item' column has beef or crop types in it

'Fed or Used for Bedding (t)' gives home grown feed

'Beef Crop Use %' gives how much is allocate to beef'

The plan is to make a new dataframe (or dicitonary) which collects all the info we need for homegrown and imported feed and animal types

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import requests

from ExtractJASManureHandlingData import map_df,nuts_df #for mapping
from ExtractAgreCalcHousing import add_prop_item, calc_weighted_mean,itl_scot


import textwrap
save_dir='../output/Diet/'
agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20251014-SAGBeef_LAD.csv'


df_raw=pd.read_csv(agrecalc_path)
df_raw=df_raw.drop(columns=['Housed (%)'])
df_beef=df_raw[df_raw['Sector']=='Beef'] #only beef rows
beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)


years=list(set(df_beef['Year End'])) #unique years
LADS= list(set(df_beef['LAD_CODE'])) #unique local authroury district codes
NUTS2=list(set(df_beef['NUTS2']))
NUTS2=[n for n in NUTS2 if n in itl_scot] #filter down to scotland.

###### Construct df for Home grown feed
#Enterprise sector items for rows to do with diet are those which have >0 in column 'Fed or Used for Bedding (t)' 
#COntruct a dataframe

home_grown_df=df_raw[df_raw['Fed or Used for Bedding (t)']>0].copy()
home_grown_df['Feed Quantity']=home_grown_df['Fed or Used for Bedding (t)']*home_grown_df['Beef Crop Use %']/100
home_grown_df=home_grown_df[['Report ID','Enterprise Sector Item','Feed Quantity']]
home_grown_df=home_grown_df.rename(columns={'Enterprise Sector Item':'Feed Name'}) #rename column
#home_grown_df=home_grown_df.reset_index()
home_grown_df['Feed Name']=home_grown_df['Feed Name']+'_hg' #to distinguigh between them in output
home_grown_types=list(home_grown_df['Feed Name'].unique()) 
#Save 
pd.DataFrame(home_grown_types).to_csv(save_dir+'home_grown_types.csv')


#####Bought in feed####
'''
##Test to see if feed is always the same regardless of beef animal type
feed_quantaties={}
bad_reports=[]
for report_id, g in df_beef.groupby(['Report ID']):
    inner_dict={}
    for feed_name,gg in g.groupby(['Feed Name']):
        inner_dict[feed_name]=gg['Feed Quantity'].nunique() #number of DM weights fed to animals in this feed name
        if inner_dict[feed_name]>1:
            bad_reports.append(gg[['Report ID','Enterprise Sector Item','Feed Type','Feed Name','Feed Quantity']])
    feed_quantaties[report_id[0]]=inner_dict  

#What we have learnt from this is farmers use the same 'Feed Name' to report multiple feeds. For example 'Minerals an vitamins.
#However, all animals seem to have the same amount of feef reported so has to be treated as a whole farm. 
#To deal with this we need to add any duplicates together and treat it as one output.  
'''

##Make df of same structre as home_grown_df for bought in feed
list_of_bought_crop=[] 
for report_id,g in df_beef.groupby(['Report ID']):
    #count cows in report.
    
    g=g.drop_duplicates(subset=['Feed Name','Feed Quantity'],keep='first') #this duplicates of tuple gets rid of the entry per cow type, but will keep feed names with different quantaties
    #now sum any feed names which are the same
    result=g.groupby('Feed Name',as_index=False)['Feed Quantity'].sum()
    result['Report ID']=[report_id[0]]*len(result)
    result['Enterprise Item']=g['Enterprise Item'].iloc[0] #also save Enterprise item, assum first
    list_of_bought_crop.append(result)

bought_in_df=pd.concat(list_of_bought_crop) #a dat
bought_in_df=bought_in_df.rename(columns={'Feed Name':'Feed Name','Feed Quantity':'Feed Quantity'})    
bought_in_types=list(bought_in_df['Feed Name'].unique())
pd.DataFrame(bought_in_types).to_csv(save_dir+'bought_in_types.csv')

all_types=home_grown_types+bought_in_types

##We now have 2 dfs for the tonnes DM of each bought in and home-growbn df
#Count cows in each report for averaging
report_id_head={} #this will stay as a dict as usesful look up.
for report_id,g1 in df_beef.groupby(['Report ID']):
    g1=g1.drop_duplicates(subset=['Enterprise Sector Item']) #becasue there are multiple rows per cow type
    total=g1['Average number over 12 Months'].sum()
    report_id_head[report_id[0]]=total

####Now data is in good format to work out avergage feed!####
#home_grown_df does not have info on 'Enterprise Info
enterprise_items=bought_in_df['Enterprise Item'].unique()
out_dict={} #{'Enterpirse item: {feed1:prop,feed2:prop....}}
for e in enterprise_items:
    e_bought_in_df=bought_in_df[bought_in_df['Enterprise Item']==e]
    e_report_ids=list(e_bought_in_df['Report ID'].unique()) #list of report ids which are this enterprise type
    e_home_grown_df=home_grown_df[home_grown_df['Report ID'].isin(e_report_ids)] #filter to only those in this enterprise type.
    e_bought_in_df=e_bought_in_df.drop(columns=['Enterprise Item']) #drop this column as we no longer need it and so they both have the same column names
    #Make one big df
    e_df=pd.concat([e_home_grown_df,e_bought_in_df])
    e_df['n_cattle']=e_df['Report ID'].apply(lambda id:report_id_head[id]) #so each row has the heads using the dict
    #Get total proportion of cows on each farm
    e_df_unique=e_df.drop_duplicates(subset='Report ID') #becuase there are duplicates per crop
    e_total_head=e_df_unique['n_cattle'].sum()
    e_df['prop_head']=e_df['n_cattle']/e_total_head #get proportion of cattle on this farm
    
    inner_dict={}
    for c in all_types: #loop thorugh crops.
        e_c_df=e_df[e_df['Feed Name']==c] #filter to crop
        c_mean,c_std=calc_weighted_mean(e_c_df,'Feed Quantity','prop_head')
        #c_mean=np.mean(np.array(e_c_df['Feed Quantity']))
        if c_mean>0: #only keep non-zero
            inner_dict[c]=c_mean
    #normalise
    tot=np.sum(list(inner_dict.values()))
    inner_dict_norm={k:v/tot for k,v in inner_dict.items()}
    out_dict[e]=inner_dict_norm


for e in out_dict.keys():
    ex=out_dict[e]
    
    plt.bar(ex.keys(),[v*100 for v in ex.values()])
    plt.xticks(rotation=45, ha='right',fontsize=5)
    plt.tight_layout()
    plt.ylabel('Percent DM diet')
    plt.title(e)
    plt.savefig(save_dir+str(e)+'_diet_bar.png',dpi=300)
    plt.close()



###Extra tests 

'''
### test if Report ID maps to exactly one 'Enterprise Item'
test_enterprise=df_beef.drop_duplicates(subset=['Report ID','Enterprise Item'],keep='first')[['Report ID', 'Enterprise Item']] #only keep unique tuples
unique_counts = test_enterprise.groupby('Report ID')['Enterprise Item'].nunique().reset_index()
unique_counts.rename(columns={'Enterprise Item': 'UniqueValueCount'}, inplace=True)
unique_counts['IsUnique'] = unique_counts['UniqueValueCount'] == 1
non_unique_enterprise=unique_counts[unique_counts['UniqueValueCount']>1]['Report ID']
'''
    