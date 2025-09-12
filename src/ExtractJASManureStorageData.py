# -*- coding: utf-8 -*-
'''
To work out what percentage of slurry is stored as x type and what perctnage of solid is stored as j type
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import geopandas as gpd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from ExtractJASManureHandlingData import  beef_items, manure_storage_items, solid_manure_items, liquid_manure_items, df, map_df, converter_df, add_proprtion_of_items,calc_weighted_mean,LADS
import textwrap
from copy import deepcopy



manure_storage_items_dict={m[0]:m[1] for m in manure_storage_items} #for easier use.


def nan_divide(col1,col2): #for making sure A/B returns nan when B=0 in pandas
    return np.where(col2 == 0, np.nan, col1 / col2)

#df is the main data frame
#map_df is used for mapping.

if __name__=="__main__":
    beef_farm_cutoff=10# how many beef cows required to be called a beef farm
    save_dir='../output/ManureStorageType/'
    
    
    df_beef_cols=df[[b[0] for b in beef_items]] #only beef item cols
    df_beef=df[df_beef_cols.sum(axis=1)>beef_farm_cutoff] #only include farms which have more than x  beef cows in total.
    print("{} out of {} farms in scotland are beef farms with this rule".format(len(df_beef),len(df)))    
    
    
    #Filter out farms that didn't fill in this sectins
    df_beef_clean=df_beef[~df_beef[[m[0] for m in manure_storage_items]].isna().all(axis=1)] #the whole row is nan
    print("Removed {} farms since they didn't fill in manure porportion sectiion".format(len(df_beef)-len(df_beef_clean)))
    filt1len=len(df_beef_clean)
    #Filter rows which have answered >100% manure_storgae
    manure_cols=df_beef_clean[[m[0] for m in manure_storage_items]]
    df_beef_clean=df_beef_clean[manure_cols.sum(axis=1)<=100]
    print("Removed {} farms since they reported more than 100% manure storage percentage".format(filt1len-len(df_beef_clean)))
    
    #remove rows that don't have any solid or lqiued
    filt2len=len(df_beef_clean)
    df_beef_clean=df_beef_clean[df_beef_clean[solid_manure_items+liquid_manure_items].sum(axis=1)>0]
    print("Removed {} farms since they had a total of 0 liquid and solid manure".format(filt2len-len(df_beef_clean)))
    #Add in total beef_numbers to each row
    df_beef_clean['beef_numbers']=df_beef_clean[[x[0] for x in beef_items]].sum(axis=1)
    
 
    #Normalise solid/liquid types so that all the solids sum to 100 and all the liquids sum to 100 with new columns
    df_beef_clean['solid_total']=df_beef_clean[solid_manure_items].sum(axis=1)
    for m in solid_manure_items:
        df_beef_clean[m+'_normed']=df_beef_clean[m]/df_beef_clean['solid_total'] 
        
    df_beef_clean['liquid_total']=df_beef_clean[liquid_manure_items].sum(axis=1)
    for m in liquid_manure_items:
        df_beef_clean[m+'_normed']=df_beef_clean[m]/df_beef_clean['liquid_total']
    
    
    
    #First we analyse the whole country starting with solid.
    #We need to only include farms which hace some solid
    df_solid=df_beef_clean[df_beef_clean['solid_total']>0].copy()
    df_solid=add_proprtion_of_items(df_solid, [b[0] for b in beef_items],prefix='beef')
    out_solid={} 
    n_farms=len(df_solid)
    #calcualte how many cows included in calculation
    n_cattle=df_solid.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
    for m in solid_manure_items:
        weighted_mean,weighted_std=calc_weighted_mean(df_solid,m+'_normed','beef_prop')

        out_solid[manure_storage_items_dict[m]]={'mean':weighted_mean,'std':weighted_std,'n-farm':n_farms,'n-cattle':n_cattle}
        
    #save out csv    
    df_out_solid=pd.DataFrame.from_dict(out_solid)
    df_out_solid.to_csv(save_dir+'solid_manure_means.csv')    
    
    #now liquid
    df_liquid=df_beef_clean[df_beef_clean['liquid_total']>0].copy()
    df_liquid=add_proprtion_of_items(df_liquid, [b[0] for b in beef_items],prefix='beef')
    n_farms=len(df_liquid)
    n_cattle=df_liquid.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
    out_liquid={} 
    for m in liquid_manure_items:
        weighted_mean,weighted_std=calc_weighted_mean(df_liquid,m+'_normed','beef_prop')
        
       
        #weighted_mean,weighted_std=calc_weighted_mean(df_liquid, m+'_normed','beef_prop')
        out_liquid[manure_storage_items_dict[m]]={'mean':weighted_mean,'std':weighted_std,'n-farm':n_farms,'n-cattle':n_cattle}
    
    df_out_liquid=pd.DataFrame.from_dict(out_liquid)
    df_out_liquid.to_csv(save_dir+'liquid_manure_means.csv')    
    
    
    
    
    
    '''
    ##Spatial analysi for solid
    out_solid_region={} #{LAD:{manure_group:percenatge}}
    counter=0
    for LAD in LADS:
        print("on {} out of {} for solid".format(counter,len(LADS)))
        df_lad=df_beef_clean[df_beef_clean['LAD22CD']==LAD] #this particular LAD
        inner_dict={}
        for m in solid_manure_items:
            df_for_calc=df_lad[~df_lad[m+'_normed'].isna()] #remove nans in this manure type. 
            #df_for_calc=deepcopy(df_beef_clean)
            df_for_calc_prop=add_proprtion_of_items(df_for_calc, [b[0] for b in beef_items],prefix='beef')
            #We need to make sure each solid 
            if len(df_for_calc_prop)==0: #if no data here
                weighted_mean=np.nan
            else:
                weighted_mean=df_for_calc_prop.apply(lambda row: row[m+'_normed']*row['beef_prop'],axis=1).sum() #this assumes all animals produce the same amount of managed manure.
            #weighted_std=np.sqrt(df_for_calc_prop.apply(lambda row: (row[m+'_normed']-weighted_mean)**2*row['beef_prop'],axis=1).sum())
            #n_farms=len(df_for_calc_prop)
            #calcualte how many cows included in calculation
            #n_cattle=df_for_calc_prop.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
            inner_dict[m]=weighted_mean*100 #make it percent        
        out_solid_region[LAD]=inner_dict
        counter+=1
        
    #join this data onto map_df    

    fig,axs=plt.subplots(nrows=int(np.ceil(len(solid_manure_items)/2)),ncols=2, figsize=(1 * len(solid_manure_items), 9), constrained_layout=True)
    axs=axs.flatten()
    #I want to get a global vmax and min so extrac data first
    all_values=[]
    for m in solid_manure_items:
        all_values += [v[m] for v in out_solid_region.values()]
    
    vmin = min(all_values)
    vmax = max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'
    xmin, ymin, xmax, ymax = map_df.total_bounds #bounds of map
    
    for i,m in enumerate(solid_manure_items):
        all_values += [v[m] for v in out_solid_region.values()]
        mapper_dict={k:v[m] for k,v in out_solid_region.items()} #to join
        map_df[m+'_mean']=map_df['LAD22CD'].map(mapper_dict)
        #plot
        
        map_df.plot(column=map_df[m + '_mean'], cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
        
    
        #axs[i].set_aspect('auto')
        axs[i].axis('off')
        wrapped_title = textwrap.fill(manure_storage_items_dict[m], width=40)
        axs[i].set_title(wrapped_title,fontsize=7)
    
    axs[-1].axis('off')     #had not needed AXIS                
    # Shared colorbar
    #fig.subplots_adjust(right=0.8) #space for colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label('% liquid manure stored as titled type', fontsize=10)
    
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0, right=0.9)  # wspace controls space between plots
    plt.savefig(save_dir+'solid_manure_regional.png',dpi=300)      
    '''
    '''
    #Spatial analysis for liquid.
    out_liquid_region={} #{LAD:{manure_group:percenatge}}
    counter=0
    for LAD in LADS:
        print("on {} out of {} for liquid".format(counter,len(LADS)))
        df_lad=df_beef_clean[df_beef_clean['LAD22CD']==LAD] #this particular LAD
        inner_dict={}
        for m in liquid_manure_items:
            df_for_calc=df_lad[~df_lad[m+'_normed'].isna()] #remove nans in this manure type. 
            #df_for_calc=deepcopy(df_beef_clean)
            df_for_calc_prop=add_proprtion_of_items(df_for_calc, [b[0] for b in beef_items],prefix='beef')
            #We need to make sure each liquid 
            if len(df_for_calc_prop)==0: #if no data here
                weighted_mean=np.nan
            else:
                weighted_mean=df_for_calc_prop.apply(lambda row: row[m+'_normed']*row['beef_prop'],axis=1).sum() #this assumes all animals produce the same amount of managed manure.
            #weighted_std=np.sqrt(df_for_calc_prop.apply(lambda row: (row[m+'_normed']-weighted_mean)**2*row['beef_prop'],axis=1).sum())
            #n_farms=len(df_for_calc_prop)
            #calcualte how many cows included in calculation
            #n_cattle=df_for_calc_prop.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
            inner_dict[m]=weighted_mean*100 #make it percent        
        out_liquid_region[LAD]=inner_dict
        counter+=1
        
    #join this data onto map_df    

    fig,axs=plt.subplots(nrows=int(np.ceil(len(liquid_manure_items)/2)),ncols=2, figsize=(2 * len(liquid_manure_items), 9), constrained_layout=True)
    axs=axs.flatten()
    #I want to get a global vmax and min so extrac data first
    all_values=[]
    for m in liquid_manure_items:
        all_values += [v[m] for v in out_liquid_region.values()]
    
    vmin = min(all_values)
    vmax = max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'
    xmin, ymin, xmax, ymax = map_df.total_bounds #bounds of map
    
    for i,m in enumerate(liquid_manure_items):
        all_values += [v[m] for v in out_liquid_region.values()]
        mapper_dict={k:v[m] for k,v in out_liquid_region.items()} #to join
        map_df[m+'_mean']=map_df['LAD22CD'].map(mapper_dict)
        #plot
        
        map_df.plot(column=map_df[m + '_mean'], cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
        
    
        #axs[i].set_aspect('auto')
        axs[i].axis('off')
        wrapped_title = textwrap.fill(manure_storage_items_dict[m], width=30)
        axs[i].set_title(wrapped_title,fontsize=7)
    
    axs[-1].axis('off')     #had not needed AXIS                
    # Shared colorbar
    #fig.subplots_adjust(right=0.8) #space for colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label('% liquid manure stored as titled type', fontsize=10)
    
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0, right=0.9)  # wspace controls space between plots
    plt.savefig(save_dir+'liquid_manure_regional.png',dpi=300)       
    '''    
        
        
  
    