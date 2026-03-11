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
from ExtractAgreCalcHousing import agrecalc_path,housed_manure_cols
from global_data import itl_scot

sys.path.append('./mappings')
from crop_groups import crop_groups, DM_yield
from livestock_units import AC_LU_mapping #to convert to livestock units for diet calculations

def feed_per_cow_fun(row): #To deal with division by zero later
    num = row['Feed Quantity_DM']
    denom = row['n_lu'] #choose here either n_lu ot n_head depending on if we want per livestock unit or per head. Livestock unit is better as it accounts for different sizes of animals, but per head is easier to understand.
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
df_beef['Housed (%)']=df_beef[housed_manure_cols].sum(axis=1) #Make new row. We use this later to get housed days..

beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)

#Make up of enterprise items proprtionally
enterprise_proportions=df_beef['Enterprise Item'].value_counts(normalize=True) #proportion of each enterprise item in the beef data. This is useful to know when we are looking at the results as it gives us an idea of how much data we have for each enterprise item.


#How many reports have more than one beef enterprise
beef_report_counts=df_beef.groupby('Report ID')['Enterprise Item'].nunique()
#What proportion have omore than one enterprise item?
proportion_multi_enterprise=(beef_report_counts>1).sum()/len(beef_report_counts)



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

#Count cows in each report for averaging and housed cows
report_id_lu={} #this will stay as a dict as usesful look up. its actually livestock unit
report_id_head={} #this is the actual head count, which we can use to check the livestock unit counts.
report_id_housed_percent={} #proportion of year housed for each report id. 
for report_id,g1 in df_beef.groupby(['Report ID']):
    g1=g1.drop_duplicates(subset=['Enterprise Sector Item']) #becasue there are multiple rows per cow type
    # livestock unit weights
    lu = g1['Enterprise Sector Item'].map(AC_LU_mapping) #makes sure and counts with livestock units
    weights = g1['Average number over 12 Months'] * lu
    total = weights.sum()
    report_id_lu[report_id[0]]=total
    report_id_head[report_id[0]]=g1['Average number over 12 Months'].sum() 
    #save housed proportion using LU weighted mean
    
    report_id_housed_percent[report_id[0]]=np.average(g1['Housed (%)'], weights=weights)

####Now data is in good format to work out avergage feed!####
#home_grown_df does not have info on 'Enterprise Info
enterprise_items=bought_in_df['Enterprise Item'].unique()
#REMOVE THE NAN ENTERPRISE
enterprise_items=np.array([x for x in enterprise_items if not (x is np.nan or (isinstance(x, str) and x.lower() == 'nan') or (isinstance(x, float) and math.isnan(x)))])


out_dict={} #{'Enterpirse item: {feed1:prop,feed2:prop....}}
out_mass={}
mean_house_percent={}
for e in enterprise_items:
    e_bought_in_df=bought_in_df[bought_in_df['Enterprise Item']==e]
    e_report_ids=list(e_bought_in_df['Report ID'].unique()) #list of report ids which are this enterprise type
    e_home_grown_df=home_grown_df[home_grown_df['Report ID'].isin(e_report_ids)] #filter to only those in this enterprise type.
    e_bought_in_df=e_bought_in_df.drop(columns=['Enterprise Item']) #drop this column as we no longer need it and so they both have the same column names
    #Make one big df
    e_df=pd.concat([e_home_grown_df,e_bought_in_df])
    e_df['n_lu']=e_df['Report ID'].map(report_id_lu) #so each row has the heads using the dict which accounts for livestock units. 
    e_df['n_head']=e_df['Report ID'].map(report_id_head) #so each row has the heads using the dict which accounts for livestock units.
    e_df['Housed (%)']=e_df['Report ID'].map(report_id_housed_percent) #so each row has the housed prop using the dict which accounts for livestock units.
    #Get total proportion of cows on each farm
    e_df_unique=e_df.drop_duplicates(subset='Report ID') #becuase there are duplicates per crop
    e_total_head=e_df_unique['n_lu'].sum()
    e_df['prop_lu']=e_df['n_lu']/e_total_head #get proportion of cattle on this farm
    e_df['Feed_per_cow'] = e_df.apply(feed_per_cow_fun, axis=1) #divides safley
    
    
    #save hosuif=ng percent out.
    house_mask= e_df_unique['Housed (%)'].notna()
    mean_house_percent[e]=np.average(e_df_unique.loc[house_mask,'Housed (%)'], weights=e_df_unique.loc[house_mask,'n_lu']) #get the mean housed percent for this enterprise type, weighted by number of cattle. This is useful to know when looking at results as it gives us an idea of how much the animals are housed on average for this enterprise type. We use the unique df here to avoid double counting farms with multiple crops.
    #Need to account for zero entries of the crop.
    #use pivot table to make the feeds columns and fiill missing with zero
    pivot = e_df.pivot_table(
         index=['Report ID', 'prop_lu'],
         columns='Feed Name',
         values='Feed_per_cow',
         aggfunc='sum',
     ).fillna(0).reset_index()

    for feed in all_types: #fill in missing feeds in this enterprise with zero
        if feed not in pivot.columns:
            pivot[feed] = 0
    inner_dict = {}
    for c in all_types:
        feed_per_cow = pivot[c]
        feed_per_housed_day = pivot[c]
        weights = pivot['prop_lu']
        
        mask1 = ~np.isnan(feed_per_cow)
        inner_dict[c] = np.average(feed_per_cow[mask1],weights=weights[mask1]) #deals with nans  
    #Normalise.
    tot = np.sum(list(inner_dict.values()))
    inner_dict_norm = {k: v / tot if tot > 0 else 0 for k, v in inner_dict.items()}
       
    #Save
    out_mass[e] = inner_dict
    out_dict[e] = inner_dict_norm



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


#Make a out per housed day version
out_mass_per_housed_day={} #copy the dict
for e,v in out_mass.items():
    inner_dict={crop:mass/(mean_house_percent[e]*365/100) for crop,mass in v.items()}
    out_mass_per_housed_day[e]=inner_dict
###Analayssis using indiviual crops.  
##    mean_house_percent


# Compute the maximum value across all plots
y_max = max(max(ex.values()) for ex in out_dict_ordered.values()) * 100
fig, axes = plt.subplots(nrows=len(out_dict_ordered), ncols=1, figsize=(14, 5 * len(out_dict_ordered)))
for ax, ex in zip(axes, out_dict_ordered.keys()):
    ex=out_dict_ordered[ex]
    ex= {k: v for k, v in ex.items() if v > 1e-4}  # remove close to zero entries to make plot clearer. We will add them back in later when we do the grouped version.
    ex=dict(sorted(ex.items(), key=lambda item: item[1], reverse=True)) #order by value sort biggest to smallet
    colors = ['green' if k.split('_')[-1]=='hg'  else 'blue' for k in ex.keys()]
    ax.bar(ex.keys(),[v*100 for v in ex.values()], color=colors)
    ax.set_xticks(range(len(ex.keys())))
    ax.set_xticklabels(ex.keys(), rotation=45, ha='right', fontsize=10)
    for tick, k in zip(ax.get_xticklabels(), ex.keys()):
        if k.split('_')[-1]=='hg':
            tick.set_color('green')
    ax.set_ylim(0, y_max)
    ax.set_ylabel('Percent DM diet')
    ax.set_title(ex,fontweight='bold')
plt.tight_layout(pad=3.0)
plt.savefig(save_dir+'combined_diet_bar.png',dpi=150,bbox_inches='tight')
plt.close()

###Save excel
pd_dict={'Enterprise Item':[],'Crop Name':[], 'LU-Weighted Mean Proportion':[], 'LU-Weighted Mean Mass (t/LU)':[], 'LU-Weighted Mean Mass per housed day (t/LU/day)':[], 'LU-Weighted Mean Housed Percent':[]}


for e,v in out_dict_ordered.items():
   pd_dict['Enterprise Item']=pd_dict['Enterprise Item']+[e]*len(v)
   pd_dict['Crop Name']+=list(v.keys())
   pd_dict['LU-Weighted Mean Proportion']+=list(v.values()) 
   pd_dict['LU-Weighted Mean Mass (t/LU)']+=[out_mass[e][crop] for crop in v.keys() ]
   pd_dict['LU-Weighted Mean Mass per housed day (t/LU/day)']+=[out_mass_per_housed_day[e][crop] for crop in v.keys() ]
   pd_dict['LU-Weighted Mean Housed Percent']+=[mean_house_percent[e]] * len(v)

out_df_all_crop=pd.DataFrame(pd_dict)
out_df_all_crop.to_excel(save_dir+'Feed_Percentages.xlsx',index=False)       

#############GROUPINGS##############

#### Use crop_groups.py to get them into more useful groups.
out_dict_grouped={} #
out_mass_grouped={}
out_mass_per_housed_day_grouped={}
for e in out_dict_ordered.keys():
    new_inner_dict={group:0 for group in crop_groups.keys()} #initialise
    mass_inner_dict={group:0 for group in crop_groups.keys()} #initialise
    out_mass_per_housed_day_inner_dict={group:0 for group in crop_groups.keys()} #initialise
    for group,list_ct in crop_groups.items():
        for ct,percent in out_dict_ordered[e].items(): #loop through all crops.
            if ct in list_ct:
                new_inner_dict[group]+=percent
                mass_inner_dict[group]+=out_mass[e][ct]
                out_mass_per_housed_day_inner_dict[group]+=out_mass_per_housed_day[e][ct]
    out_dict_grouped[e]=new_inner_dict            
    out_mass_grouped[e]=mass_inner_dict
    out_mass_per_housed_day_grouped[e]=out_mass_per_housed_day_inner_dict
#plot

y_max_group = max(max(ex.values()) for ex in out_dict_grouped.values()) * 100
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()  # Flatten to iterate easily

for i, (e, ex) in enumerate(out_dict_grouped.items()):
    ax = axes[i]
    
    # Extract keys and convert values to percentages
    labels = list(ex.keys())
    values = [v * 100 for v in ex.values()]
    
    # Create the bar plot
    ax.bar(labels, values, color='skyblue', edgecolor='navy')
    
    # Styling and Labels
    ax.set_title(f"Grouped: {e}", fontsize=14, fontweight='bold')
    ax.set_ylabel('Percent DM diet')
    ax.set_ylim(0, y_max_group)
    
    # Format X-axis ticks
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)

# Remove any unused subplots (if dict length < 8)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust spacing to prevent title/label overlap
plt.tight_layout(pad=5.0)

# Save the combined figure
plt.savefig(save_dir + 'feed_grouping/ALL_GROUPED_diet_bars.png', dpi=300)

#Save excel

pd_group_dict={'Enterprise Item':[],'Group Name':[], 'LU-Weighted Mean Proportion':[], 'LU-Weighted Mean Mass (t/LU)':[], 'LU-Weighted Mean Mass per housed day (t/LU/day)':[],'LU-Weighted Mean Housed Percent':[]}

for e,v in out_dict_grouped.items():
    pd_group_dict['Enterprise Item']+=[e]*len(v)
    pd_group_dict['Group Name']+=list(v.keys())
    pd_group_dict['LU-Weighted Mean Proportion']+=list(v.values()) 
    pd_group_dict['LU-Weighted Mean Mass (t/LU)']+=[out_mass_grouped[e][group] for group in v.keys() ]
    pd_group_dict['LU-Weighted Mean Mass per housed day (t/LU/day)']+=[out_mass_per_housed_day_grouped[e][group] for group in v.keys() ]
    pd_group_dict['LU-Weighted Mean Housed Percent']+=[mean_house_percent[e]] * len(v)
    
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
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for i, (e, v) in enumerate(enterprise_make_up.items()):
    ax = axes[i]
    total = sum(v.values())
    normalized_data = {k: (vv / total) * 100 for k, vv in v.items()}
    
    labels = list(normalized_data.keys())
    values = list(normalized_data.values())
    
    ax.bar(labels, values, color='forestgreen', edgecolor='black')
    
    # Styling
    ax.set_title(f"Enterprise: {e}", fontsize=14, fontweight='bold')
    ax.set_ylabel('Percent of animals')
    ax.set_ylim(0, 100) # Since it's normalized, 100% is the logical max
    
    # Format X-axis
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust spacing
plt.tight_layout(pad=5.0)

# Save the combined figure
plt.savefig(save_dir + 'enterprise_make_up/ALL_ENTERPRISE_MakeUp.png', dpi=300)


   
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
 
plt.close('all')
 
                          

    