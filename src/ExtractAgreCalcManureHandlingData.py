'''
Code to process AgreCalc and pick out manure management percentages of solid vs liquid for beef

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
import warnings
#from pandas.errors import SettingWithCopyWarning
#warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import requests


from ExtractAgreCalcHousing import add_prop_item
from global_data import itl_scot, map_df,nuts_df, calc_weighted_mean


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

#Add in solid percents
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



#Make spatial means
plotting = 'LAD'
geoplotting={'nuts':{'shapes':NUTS2,'col':'NUTS2','df':nuts_df,'map_col':'ITL225CD'},
                 'LAD':{'shapes':LADS,'col':'LAD_CODE','df':map_df,'map_col':'LAD22CD'}
                        } #to format plotting below


##First extract means for each region and beef type
df_beef_geo=df_beef_clean[~df_beef_clean[geoplotting[plotting]['col']].isna()] #drop the rows with no lads code
rows3=len(df_beef_geo)
farm_years3=len(list(set(df_beef_geo['Report ID'])))
print('Dropped {} farm-years since there was no valid postcode'.format(farm_years2-farm_years3))
out_liquid_region={} #{LAD:{bt:{housed:housed...,n_cattle:}}}
for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
    df_lad=df_beef_geo[df_beef_geo[geoplotting[plotting]['col']]==LAD]
    inner_dict={}
    for k,v in beef_groups.items():
        df_lad_bt=add_prop_item(df_lad,v)# add in proportions. Shortens df down to only the bt rows
        n_farms = len(list(set(df_lad_bt['Report ID'])))
        n_cattle = df_lad_bt['Average number over 12 Months'].sum()
        mean,std=calc_weighted_mean(df_lad_bt,'liquid_percent','prop')
        d={'n_farms':n_farms,'n_cattle':n_cattle,'liquid_mean':mean,'liquid_std':std}
        inner_dict[k]=d
    out_liquid_region[LAD]=inner_dict

#Join dcitionary onto df and plot
key_to_plot='liquid_mean'
fig,axs=plt.subplots(nrows=1,ncols=2, figsize=(10, 5), constrained_layout=True)
axs=axs.flatten() #an axs for each beef type
cmap = 'viridis'

#To normalise color map 0-100
norm = Normalize(vmin=0, vmax=100)

gdf=geoplotting[plotting]['df'] #pick it out for ease of syntax
for i,bt in enumerate(list(beef_groups.keys())): #bt=<1 ot >1 here
    mapper_dict={k:v[bt][key_to_plot] for k,v in out_liquid_region.items()} #to join
    
  
    gdf[bt+'_mean']=gdf[geoplotting[plotting]['map_col']].map(mapper_dict)
    #gdf=gdf.dropna(subset=[bt+'_mean'])
    gdf_plot=gdf.plot(column=bt + '_mean', cmap=cmap,ax=axs[i],norm=None, legend=True,missing_kwds={'color': 'lightgrey'})
    #axs[i].set_aspect('auto')
    axs[i].axis('off')
    wrapped_title = textwrap.fill(bt, width=40)
    axs[i].set_title(wrapped_title,fontsize=14)
    colorbar = gdf_plot.get_figure().get_axes()[1]  # second axis is the colorbar
    colorbar.set_ylabel('% Manure managed as liquid', fontsize=12)
    
#plt.show()
axs[-1].axis('off')
axs[-2].axis('off')
fig.savefig(save_dir+key_to_plot+'{}_spatial.png'.format(geoplotting[plotting]['col']),dpi=300)
plt.close('all')

#Time plots (tyring groubby method)
#make new column which says if its <1 or >1
beef_groups_inv={v: k for k, values in beef_groups.items() for v in values}
df_beef_clean['age_group']=df_beef_clean["Enterprise Sector Item"].apply(lambda row: beef_groups_inv[row])
#get rid of duplicates per feed problem:
df_beef_time=df_beef_clean.drop_duplicates(subset=["Enterprise Sector Item", "Report ID"])

def helper_liquid_solid_stats(group,liquid_solid="liquid_percent"):
    n_cattle=group["Average number over 12 Months"].sum()
    group["prop"]=group["Average number over 12 Months"]/n_cattle #add in proportions
    n_farms=len(list(set(group["Report ID"])))
    mean, std = calc_weighted_mean(group, liquid_solid, "prop")
    return {'n_farms':n_farms,'n_cattle':n_cattle,liquid_solid+'_mean':mean,liquid_solid+'_std':std}



df_grouped=df_beef_time.groupby(["age_group","Year End"])

out_liquid=df_grouped.apply(lambda g: helper_liquid_solid_stats(g,liquid_solid="liquid_percent")).to_dict()


##make plots
years=list(set([k[1] for k in out_liquid.keys()]))
age_groups=list(set([k[0] for k in out_liquid.keys()]))
for age_group in age_groups: #loop overage groups
    means=[]
    stds=[]
    for y in set([k[1] for k in out_liquid.keys()]):
        means.append(out_liquid[(age_group,y)]['liquid_percent_mean'])
        stds.append(out_liquid[(age_group,y)]['liquid_percent_std'])
    plt.plot(years,means,label=age_group)
plt.xlabel('Year')
plt.ylabel('% managed manure as liquid')
plt.legend()
plt.savefig(save_dir+'liquid_time_trends',dpi=200)







