'''
Code to process JAC2023 and pick out manure management percentages of solid vs liquid for beef

IN JAC anlaysis Solid was 'Manure solid storage in heaps %','Manure stored in compost piles %', 'Manure stored in pits below animal confinement %', 'Manure stored in deep litter systems,
'Manure stored in deep litter systems %', 'Manure stored in other facilities (not elsewhere classified) %'

And liquid: 'Liquid manure/slurry storage without cover %', 'Liquid manure/slurry storage without cover %', 'Liquid manure/slurry storage with impermeable cover %', 'Daily spread %'

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
from ExtractAgreCalcHousing import add_prop_item, calc_weighted_mean


import textwrap

save_dir='../output/ManureHandlingSystem/AC/'
agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20250728-SAGBeef_LAD.csv'

df_raw=pd.read_csv(agrecalc_path)
df_raw=df_raw.drop(columns=['Housed (%)'])
df_beef=df_raw[df_raw['Sector']=='Beef'] #only beef rows
beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)

#define older than 1 year and less than 
beef_groups={'<1 Yr':['Entire 0-12 mnth','Heifer  0-12 mnth','Steer  0-12 mnth'],
             '>1 Yr':['Bull','Entire 12-24 mnth','Heifer 12-24 mnth','Heifer 24-36 mnth','Steer  24-36 mnth','Steer 12-24 mnth','Suckler cow']}

years=list(set(df_beef['Year End'])) #unique years
LADS= list(set(df_beef['LAD_CODE'])) #unique local authroury district codes

#Define  manure types
manure_cols=['Pasture (%)', 'Hill ground (%)', 'Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)','Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']
housed_manure_cols=['Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)', 'Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']
liquid_cols=['Liquid Slurry (%)','Pit storage (Slats) (%)']
solid_cols=['Solid storage (FYM) (%)','Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']


#drop rows who's manure percentages don't add to 100.
farm_years1=len(list(set(df_beef['Report ID'])))
print('Started with {} many farm years'.format(farm_years1))
rows1=len(df_beef)
print('Started with {} rows'.format(rows1))
df_beef_clean=df_beef[df_beef[manure_cols].sum(axis=1)==100]
rows2=len(df_beef_clean)
farm_years2=len(list(set(df_beef_clean['Report ID'])))
print('Dropped {} rows since thier manure percentage didnt sum to 100'.format(rows1-rows2))
print('Dropped {} farm years since thier manure percentage didnt sum to 100'.format(farm_years1-farm_years2))

#Add in sums of manure type.
df_beef_clean['managed_percent']=df_beef_clean[housed_manure_cols].sum(axis=1)
#drop rows that have no managed manure
df_beef_clean= df_beef_clean[df_beef_clean["managed_percent"] != 0]
df_beef_clean['solid_percent']=df_beef_clean[solid_cols].sum(axis=1)/df_beef_clean['managed_percent']
df_beef_clean['liquid_percent']=df_beef_clean[liquid_cols].sum(axis=1)/df_beef_clean['managed_percent']

###Make histogram and spit out means and stds
out_dict={}#{<1:{mean_liquid:x,std_liquid,....,n_farm_years,n_cattle_years}
fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(10, 10))
axs=axs.flatten()
i=0
for k,v in beef_groups.items(): #loop through <1yr >1 yr
    #add in proportion of beef cows owned so we can do weighted averages
    df_beef_manure=add_prop_item(df_beef_clean, v) #this crops the datframe down a lot to only rows that repersent beef animals
    n_farm_years=len(list(set(df_beef_manure['Report ID']))) #unique reports
    n_cattle_years=df_beef_manure['Average number over 12 Months'].sum(axis=0) # total number of animals in this 
    solid_mean,solid_std=calc_weighted_mean(df_beef_manure,'solid_percent','prop')
    liquid_mean,liquid_std=calc_weighted_mean(df_beef_manure,'liquid_percent','prop')
    out_dict[k]={'liquid_mean':liquid_mean*100,'liquid_std':liquid_std*100,
                 'solid_mean':solid_mean*100,'solid_std':solid_std*100,
                 'n-enterprise-years':n_farm_years,'n-cattle-years':n_cattle_years}
    
    #need to drop out 0 weight rows for plot so we can do a densiry
    df_beef_manure=df_beef_manure[df_beef_manure['prop']>0]
    #plot liquid
    axs[i].hist(df_beef_manure['liquid_percent'],weights=df_beef_manure['prop'],density=True,bins=100)   
    axs[i].set_xlabel('Proportion of Manure',fontsize=9)
    axs[i].set_ylabel('Probability',fontsize=9)
    axs[i].tick_params(axis='both', labelsize=9)
    wrapped_title = textwrap.fill(k+' liquid_percent'+', n-enterprise-years={}'.format(n_farm_years), width=70)
    axs[i].set_title(wrapped_title,fontsize=9)
    
    #plt solid
    axs[i+1].hist(df_beef_manure['solid_percent'],weights=df_beef_manure['prop'],density=True,bins=100)
    axs[i+1].set_xlabel('Proportion of Manure',fontsize=9)
    axs[i+1].set_ylabel('Probability',fontsize=9)
    axs[i+1].tick_params(axis='both', labelsize=9)
    wrapped_title = textwrap.fill(k+' solid_percent'+', n-enterprise-years={}'.format(n_farm_years), width=70)
    axs[i+1].set_title(wrapped_title,fontsize=9)
    i=i+2
plt.savefig(save_dir+'histogram_AC.png',dpi=300)
df_out=pd.DataFrame(out_dict)
df_out=df_out[sorted(df_out.columns)] # arrange alphabeticaly
df_out.to_csv(save_dir+'liquid_vs_solid_AC.csv')
