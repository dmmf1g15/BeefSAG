'''
Code to process Agrecalc and pick out manure storage percentages of solid and liquid for beef



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
NUTS2=list(set(df_beef['NUTS2']))
NUTS2=[n for n in NUTS2 if n in itl_scot] #filter down to scotland.




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
farm_years3=df_beef_clean["Report ID"].nunique()
print('Dropped {} farm years since they had no managed manure'.format(farm_years2-farm_years3))

df_beef_clean['solid_percent']=df_beef_clean[solid_cols].sum(axis=1)/df_beef_clean['managed_percent']
df_beef_clean['liquid_percent']=df_beef_clean[liquid_cols].sum(axis=1)/df_beef_clean['managed_percent']

#Drop duplicate rows which are repeated for different feed types.
df_beef_clean=df_beef_clean.drop_duplicates(subset=["Enterprise Sector Item", "Report ID"])

#normalise each solid/ or liquid managemetn type to the percentageof manure or liquid by making new columns
df_beef_clean['solid_total']=df_beef_clean[solid_cols].sum(axis=1)
df_beef_clean['liquid_total']=df_beef_clean[liquid_cols].sum(axis=1)
for m in solid_cols:
    df_beef_clean[m+'_normed']=df_beef_clean[m]/df_beef_clean['solid_total']
for m in liquid_cols:
    df_beef_clean[m+'_normed']=df_beef_clean[m]/df_beef_clean['liquid_total']


#df_beef_clean.to_csv('test.csv')
    
##first do mean and std for solid
#only include farms whicih include some solid
df_solid=df_beef_clean[df_beef_clean['solid_total']>0]
#add in the proportion of all beef animals thats is in each row. Doesn't distinguish between animal types.
df_solid=add_prop_item(df_solid,beef_types)
n_farms=df_solid['Report ID'].nunique() #count untious farm_years
n_cattle=df_solid["Average number over 12 Months"].sum(axis=0)
out_solid={}
for m in solid_cols:
    mean,std=calc_weighted_mean(df_solid,m+'_normed','prop')
    #calculate number farms which have this type
    d={'mean':mean,'std':std,'n_enterprise_years':n_farms,'n_cattle_years':n_cattle}
    #calculate number of animals which have at least some of this this type
    out_solid[m]=d
check_sum_solid=np.sum([v['mean'] for v in out_solid.values()])
    
 
#Now do liquid    
#only include farms whicih include some liquid
df_liquid=df_beef_clean[df_beef_clean['liquid_total']>0]
#add in the proportion of all beef animals thats is in each row. Doesn't distinguish between animal types.
df_liquid=add_prop_item(df_liquid,beef_types)
n_farms=df_liquid['Report ID'].nunique() #count untious farm_years
n_cattle=df_liquid["Average number over 12 Months"].sum(axis=0)
out_liquid={}
for m in liquid_cols:
    mean,std=calc_weighted_mean(df_liquid,m+'_normed','prop')
    #calculate number farms which have this type
    d={'mean':mean,'std':std,'n_enterprise_years':n_farms,'n_cattle_years':n_cattle}
    #calculate number of animals which have at least some of this this type
    out_liquid[m]=d
check_sum_liquid=np.sum([v['mean'] for v in out_liquid.values()])





