'''
Code to process AgreCalc data and pick out diet of animal types

'Enterprise Sector Item' column has beef or crop types in it

'Fed or Used for Bedding (t)' gives home grown feed

'Beef Crop Use %' gives how much is allocate to beef'

The plan is to make a new dataframe (or dicitonary) which collects all the info we need for homegrown and imported feed and animal types

'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
import warnings
import sys

#from pandas.errors import SettingWithCopyWarning
#warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import requests
from global_data import itl_scot

sys.path.append('./mappings')
from crop_groups import crop_groups, DM_yield


def feed_per_cow(row): #To deal with division by zero later
    num = row['Feed Quantity_DM']
    denom = row['n_cattle']
    if pd.isna(denom) or denom == 0:
        return 0
    else:
        return num / denom
    

import textwrap
save_dir='../output/Diet/'
agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20251014-SAGBeef_LAD.csv'


forage_crops = ['Forage rye - Forage_hg', 'Kale /stubble turnips/ swedes - Forage_hg',
                'Legume forages (clovers, lucerne) (all) - Forage_hg','Fodder beet - Forage_hg', 'Grass - grazing '
                ] ##possibly these are the outdoot feeds.


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

home_grown_df['Feed Quantity']=home_grown_df['Fed or Used for Bedding (t)']*home_grown_df['Beef Crop Use %']/100 #Fresh matetr at the moment

home_grown_df=home_grown_df[['Report ID','Enterprise Sector Item','Feed Quantity']]
home_grown_df=home_grown_df.rename(columns={'Enterprise Sector Item':'Feed Name'}) #rename column
#home_grown_df=home_grown_df.reset_index()
home_grown_df['Feed Name']=home_grown_df['Feed Name']+'_hg' #to distinguigh between them in output

#Remove "Forage crops"
home_grown_df=home_grown_df[~home_grown_df['Feed Name'].isin(forage_crops)]

###Account for DM yield
home_grown_df['Feed Quantity_DM']=home_grown_df.apply(lambda row: DM_yield[row['Feed Name']]*row['Feed Quantity'],axis=1) #Now it is DM


home_grown_types=list(home_grown_df['Feed Name'].unique()) 
#Save 
pd.DataFrame(home_grown_types).to_csv(save_dir+'home_grown_types.csv')


########


#####Bought in feed####

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
bought_in_df=bought_in_df.rename(columns={'Feed Name':'Feed Name','Feed Quantity':'Feed Quantity'})    #This is FM
#Remove forage
bought_in_df=bought_in_df[~bought_in_df['Feed Name'].isin(forage_crops)]
###Account for DM yield
bought_in_df['Feed Quantity_DM']=bought_in_df.apply(lambda row: DM_yield[row['Feed Name']]*row['Feed Quantity'],axis=1) #Now it is DM
#drop empty enterprise reports
bought_in_df=bought_in_df.dropna(subset=['Enterprise Item'])

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
#REMOVE THE NAN ENTERPRISE
enterprise_items=np.array([x for x in enterprise_items if not (x is np.nan or (isinstance(x, str) and x.lower() == 'nan') or (isinstance(x, float) and math.isnan(x)))])


out_dict={} #{'Enterpirse item: {feed1:prop,feed2:prop....}}
out_mass={}
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
    e_df['Feed_per_cow'] = e_df.apply(feed_per_cow, axis=1) #divides safley
    
    #Need to account for zero entries of the crop.
    #use pivot table to make the feeds columns and fiill missing with zero
    pivot = e_df.pivot_table(
         index=['Report ID', 'prop_head'],
         columns='Feed Name',
         values='Feed_per_cow',
         aggfunc='sum',
         fill_value=0
     ).reset_index()
    
    inner_dict = {}
    for c in all_types:
        if c in pivot.columns:
            values = pivot[c]
        else: # this deals with the fact that not all crops appear in this enterprise. So we fill with 0. IS THIS NEEDED?
            values = pd.Series([0] * len(pivot))
       
        weights = pivot['prop_head']
        #Mask to deal with a few nans in the weights which fucked things up.
        mask = ~np.isnan(values)
        c_mean = np.average(values[mask], weights=weights[mask])
        inner_dict[c] = c_mean
       
    #Normalise.
    tot = np.sum(list(inner_dict.values()))
    inner_dict_norm = {k: v / tot if tot > 0 else 0 for k, v in inner_dict.items()}
       
    #Save
    out_mass[e] = inner_dict
    out_dict[e] = inner_dict_norm
       
    '''
    inner_dict={}
    for c in all_types: #loop thorugh crops.
        e_c_df=e_df[e_df['Feed Name']==c] #filter to crop
        c_mean,c_std=calc_weighted_mean(e_c_df,'Feed_per_cow','prop_head') #this is mean feed per cow
        inner_dict[c]=c_mean      
    #normalise
    tot=np.sum(list(inner_dict.values()))
    inner_dict_norm={k:v/tot for k,v in inner_dict.items()}
    out_mass[e]=inner_dict
    out_dict[e]=inner_dict_norm
    '''     

###Make sure each dicttionary has same keys and is same order so its easier to compare
all_crop_keys=[]
for e in enterprise_items:
    crop_keys=list(out_dict[e].keys())
    all_crop_keys=all_crop_keys+crop_keys
all_crop_keys=list(set(all_crop_keys))
all_crop_keys=sorted(all_crop_keys)

##make the dictionaries have the same keys in the same order:]
out_dict_ordered={}
for e in enterprise_items:
    new_dict={}
    for ck in all_crop_keys:
        if ck in out_dict[e].keys():
            new_dict[ck]=out_dict[e][ck]
        else:
            new_dict[ck]=0
    out_dict_ordered[e]=new_dict
        
###Analayssis using indiviual crops.  
##    

highlight_value=5
# Compute the maximum value across all plots
y_max = max(max(ex.values()) for ex in out_dict_ordered.values()) * 100
for e in out_dict_ordered.keys():
    ex=out_dict_ordered[e]
    f=plt.figure(figsize=(16, 6))
    colors = ['red' if v*100 > highlight_value else 'blue' for v in ex.values()] #highlight big vlaues
    
    plt.bar(ex.keys(),[v*100 for v in ex.values()],color=colors)
    xticks=plt.xticks(rotation=45, ha='right',fontsize=5)[1]
    for tick, v in zip(xticks, ex.values()):
        if v*100 > highlight_value:
            tick.set_color('red')
    plt.ylim(0, y_max)
    plt.tight_layout(pad=3.0)
    plt.ylabel('Percent DM diet')
    plt.title(e)
    plt.savefig(save_dir+str(e)+'_diet_bar.png',dpi=300)
    plt.close()

###Save excel
pd_dict={'Enterprise Item':[],'Crop Name':[], 'Head-Weighted Mean Proportion':[], 'Head-Weighted Mean Mass (t/head)':[]}

#with pd.ExcelWriter(save_dir+'Feed_Percentages.xlsx', engine='openpyxl') as writer:
for e,v in out_dict_ordered.items():
   pd_dict['Enterprise Item']=pd_dict['Enterprise Item']+[e]*len(v)
   pd_dict['Crop Name']+=list(v.keys())
   pd_dict['Head-Weighted Mean Proportion']+=list(v.values()) 
   pd_dict['Head-Weighted Mean Mass (t/head)']+=[out_mass[e][crop] for crop in v.keys() ]
 
out_df_all_crop=pd.DataFrame(pd_dict)
out_df_all_crop.to_excel(save_dir+'Feed_Percentages.xlsx',index=False)       

#############GROUPINGS##############

#### Use crop_groups.py to get them into more useful groups.
out_dict_grouped={} #
out_mass_grouped={}
for e in out_dict_ordered.keys():
    new_inner_dict={group:0 for group in crop_groups.keys()} #initialise
    mass_inner_dict={group:0 for group in crop_groups.keys()} #initialise
    for group,list_ct in crop_groups.items():
        for ct,percent in out_dict_ordered[e].items(): #loop through all crops.
            if ct in list_ct:
                new_inner_dict[group]+=percent
                mass_inner_dict[group]+=out_mass[e][ct]
    out_dict_grouped[e]=new_inner_dict            
    out_mass_grouped[e]=mass_inner_dict
#plot

y_max_group = max(max(ex.values()) for ex in out_dict_grouped.values()) * 100
for e in out_dict_grouped.keys():
    ex=out_dict_grouped[e]
 
    plt.bar(ex.keys(),[v*100 for v in ex.values()])
    xticks=plt.xticks(rotation=45, ha='right',fontsize=10)[1]
    plt.tight_layout(pad=3.0)
    plt.ylim(0,y_max_group)
    plt.ylabel('Percent DM diet')
    plt.title(e)
    plt.savefig(save_dir+'feed_grouping/'+str(e)+'_GROUPED_diet_bar.png',dpi=300)
    plt.close()


#Save excel

pd_group_dict={'Enterprise Item':[],'Group Name':[], 'Head-Weighted Mean Proportion':[], 'Head-Weighted Mean Mass (t/head)':[]}

for e,v in out_dict_grouped.items():
   pd_group_dict['Enterprise Item']+=[e]*len(v)
   pd_group_dict['Group Name']+=list(v.keys())
   pd_group_dict['Head-Weighted Mean Proportion']+=list(v.values()) 
   pd_group_dict['Head-Weighted Mean Mass (t/head)']+=[out_mass_grouped[e][group] for group in v.keys() ]
 
out_df_grouped=pd.DataFrame(pd_group_dict)
out_df_grouped.to_excel(save_dir+'FeedGroup_Percentages.xlsx',index=False)       


#######Work out what animal types are on each enterprise item type
#


enterprise_make_up={} #{enterpirse:{c1:count}}
for e in enterprise_items:
    inner_dict={bt:0 for bt in beef_types}# 
    df_beef_e=df_beef[df_beef['Enterprise Item']==e] #filter to thjs enterprise
    for report_id,g1 in df_beef_e.groupby(['Report ID']):
        g1=g1.drop_duplicates(subset=['Enterprise Sector Item']) #becasue there are multiple rows per cow type
        for bt in beef_types:
            val_series = g1[g1['Enterprise Sector Item'] == bt]['Average number over 12 Months']
            val = val_series.iloc[0] if not val_series.empty else 0  # get value or 0
            if pd.isna(val):
                val=0
            inner_dict[bt] += val
    enterprise_make_up[e] =inner_dict       

    
### Make plots 
#y_max_animals = max(max(ex.values()) for ex in enterprise_make_up.values()) * 100
for e,v in enterprise_make_up.items():
    
    total = sum(v.values())
    normalized_data = {k: vv / total for k, vv in v.items()}
    plt.bar(normalized_data.keys(),normalized_data.values())
    xticks=plt.xticks(rotation=45, ha='right',fontsize=10)[1]
    plt.tight_layout(pad=3.0)
    plt.ylabel('Percent of animals')
    plt.title(e)
    plt.savefig(save_dir+'enterprise_make_up/'+str(e)+'_MakeUp.png',dpi=300)
    plt.close()


   
###Extra tests 
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

'''
### test if Report ID maps to exactly one 'Enterprise Item'
test_enterprise=df_beef.drop_duplicates(subset=['Report ID','Enterprise Item'],keep='first')[['Report ID', 'Enterprise Item']] #only keep unique tuples
unique_counts = test_enterprise.groupby('Report ID')['Enterprise Item'].nunique().reset_index()
unique_counts.rename(columns={'Enterprise Item': 'UniqueValueCount'}, inplace=True)
unique_counts['IsUnique'] = unique_counts['UniqueValueCount'] == 1
non_unique_enterprise=unique_counts[unique_counts['UniqueValueCount']>1]['Report ID']
'''
'''   
##test what DM per cow is in each enterprise
DM_per_head={}
for r in bought_in_df['Report ID'].unique():
    r_BI_df=bought_in_df[bought_in_df['Report ID']==r]
    r_HG_df=home_grown_df[home_grown_df['Report ID']==r]
    
    BI_tot_DM=r_BI_df['Feed Quantity_DM'].sum()
    HG_tot_DM=r_HG_df['Feed Quantity_DM'].sum()
    DM_per_head[r]=(BI_tot_DM+HG_tot_DM)/report_id_head[r]
 
 '''
 
 
 
                          

    