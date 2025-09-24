'''
Code to process Agrecalc and pick out manure storage percentages of solid and liquid for beef



'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import geopandas as gpd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import requests

from ExtractJASManureHandlingData import map_df,nuts_df #for mapping
from ExtractAgreCalcHousing import add_prop_item, calc_weighted_mean,itl_scot, NUTS2,LADS,map_df,nuts_df


import textwrap

save_dir='../output/ManureStorageType/AC/'
agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20250728-SAGBeef_LAD.csv'

plotting = 'nuts'
geoplotting={'nuts':{'shapes':NUTS2,'col':'NUTS2','df':nuts_df,'map_col':'ITL225CD'},
                 'LAD':{'shapes':LADS,'col':'LAD_CODE','df':map_df,'map_col':'LAD22CD'}
                        } #to format plotting below

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
    
df_solid=df_beef_clean[df_beef_clean['solid_total']>0]
df_liquid=df_beef_clean[df_beef_clean['liquid_total']>0]

'''
##first do mean and std for solid
#only include farms whicih include some solid

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
out_solid_df=pd.DataFrame.from_dict(out_solid)
out_solid_df.to_csv(save_dir+'solid_manure_AC.csv')    
 
#Now do liquid    
#only include farms whicih include some liquid

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
out_liquid_df=pd.DataFrame.from_dict(out_liquid)
out_liquid_df.to_csv(save_dir+'liquid_manure_AC.csv')
'''



###Spatial plots LIQUID
df_beef_geo_liquid=df_liquid[~df_liquid[geoplotting[plotting]['col']].isna()] #drop invalid postcodes
out_liquid_region={} #{LAD:{bt:{housed:housed...,n_cattle:}}}
for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
    df_lad=df_beef_geo_liquid[df_beef_geo_liquid[geoplotting[plotting]['col']]==LAD]
    df_lad=add_prop_item(df_lad,beef_types)
    inner_dict={}
    for m in liquid_cols:
        mean,std=calc_weighted_mean(df_lad,m+'_normed','prop')
        inner_dict[m]=mean*100
    out_liquid_region[LAD]=inner_dict
    
#join this data onto map_df    
gdf=geoplotting[plotting]['df'] #pick it out for ease of syntax
fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(10, 6))
axs=axs.flatten()
#I want to get a global vmax and min so extrac data first
all_values=[]
for m in liquid_cols:
    all_values += [v[m] for v in out_liquid_region.values()]

vmin = min(all_values)
vmax = max(all_values)
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = 'viridis'
xmin, ymin, xmax, ymax = gdf.total_bounds #bounds of map

for i,m in enumerate(liquid_cols):
    all_values += [v[m] for v in out_liquid_region.values()]
    mapper_dict={k:v[m] for k,v in out_liquid_region.items()} #to join
    gdf[m+'_mean']=gdf[geoplotting[plotting]['map_col']].map(mapper_dict)
    #plot
    
    gdf.plot(column=m + '_mean', cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
    

    #axs[i].set_aspect('auto')
    axs[i].axis('off')
    wrapped_title = textwrap.fill(m, width=40)
    axs[i].set_title(wrapped_title,fontsize=11)

axs[-1].axis('off')     #had not needed AXIS                
# Shared colorbar
#fig.subplots_adjust(right=0.8) #space for colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, ax=axs)
cbar.set_label('% liquid manure stored as titled type', fontsize=16)

#plt.tight_layout()
#plt.subplots_adjust(wspace=0, right=0.9)  # wspace controls space between plots
plt.savefig(save_dir+'liquid_manure_AC_{}.png'.format(geoplotting[plotting]['col']),dpi=300)


#Spatial plots ###############SOLID
df_beef_geo_solid=df_solid[~df_solid[geoplotting[plotting]['col']].isna()] #drop invalid postcodes
out_solid_region={} #{LAD:{bt:{housed:housed...,n_cattle:}}}
for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
    df_lad=df_beef_geo_solid[df_beef_geo_solid[geoplotting[plotting]['col']]==LAD]
    df_lad=add_prop_item(df_lad,beef_types)
    inner_dict={}
    for m in solid_cols:
        mean,std=calc_weighted_mean(df_lad,m+'_normed','prop')
        inner_dict[m]=mean*100
    out_solid_region[LAD]=inner_dict
    
#join this data onto map_df    
gdf=geoplotting[plotting]['df'] #pick it out for ease of syntax
fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(7, 8))
axs=axs.flatten()
#I want to get a global vmax and min so extrac data first
all_values=[]
for m in solid_cols:
    all_values += [v[m] for v in out_solid_region.values()]

vmin = min(all_values)
vmax = max(all_values)
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = 'viridis'
xmin, ymin, xmax, ymax = gdf.total_bounds #bounds of map

for i,m in enumerate(solid_cols):
    all_values += [v[m] for v in out_solid_region.values()]
    mapper_dict={k:v[m] for k,v in out_solid_region.items()} #to join
    gdf[m+'_mean']=gdf[geoplotting[plotting]['map_col']].map(mapper_dict)
    #plot
    
    gdf.plot(column=m + '_mean', cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
    

    #axs[i].set_aspect('auto')
    axs[i].axis('off')
    wrapped_title = textwrap.fill(m, width=40)
    axs[i].set_title(wrapped_title,fontsize=11)

axs[-1].axis('off')     #had not needed AXIS                
# Shared colorbar
#fig.subplots_adjust(right=0.8) #space for colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, ax=axs)
cbar.set_label('% solid manure stored as titled type', fontsize=16)

#plt.tight_layout()
#plt.subplots_adjust(wspace=0, right=0.9)  # wspace controls space between plots
plt.savefig(save_dir+'solid_manure_AC_{}.png'.format(geoplotting[plotting]['col']),dpi=300)

'''
#In time plots
liquid_plot={m:[] for m in liquid_cols}
for y in years:
    df_year=df_liquid[df_liquid["Year End"]==y]
    df_year=add_prop_item(df_year,beef_types)
    for m in liquid_cols:
        mean,std=calc_weighted_mean(df_year,m+'_normed','prop')
        liquid_plot[m].append(mean)


for m in liquid_cols:
    plt.plot(years,liquid_plot[m],label=m)
plt.xlabel('Year')
plt.ylabel('Percent of liquid manure managed as')    
plt.legend()
plt.savefig(save_dir+'liquid_in_time_AC.png', dpi=200)

#In time plots SOLID
solid_plot={m:[] for m in solid_cols}
for y in years:
    df_year=df_solid[df_solid["Year End"]==y]
    df_year=add_prop_item(df_year,beef_types)
    for m in solid_cols:
        mean,std=calc_weighted_mean(df_year,m+'_normed','prop')
        solid_plot[m].append(mean)
plt.close('all')

for m in solid_cols:
    plt.plot(years,solid_plot[m],label=m)
plt.xlabel('Year')
plt.ylabel('Percent of solid manure managed as')    
plt.legend()
plt.savefig(save_dir+'solid_in_time_AC.png', dpi=200)
plt.close('all')
'''
