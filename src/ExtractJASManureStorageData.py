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

from ExtractJASManureHandlingData import  beef_items, manure_storage_items, solid_manure_items, liquid_manure_items,add_proprtion_of_items
from global_data import df, map_df,nuts_df,itl_scot,NUTS2,calc_weighted_mean,LADS
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
    #Switches for plotting if needed
    plotting='LAD' #chose #nuts or LAD to decide what gets plot.
    
    geoplotting={'nuts':{'shapes':NUTS2,'col':'ITL225CD','df':nuts_df,'map_col':'ITL225CD'},
                 'LAD':{'shapes':LADS,'col':'LAD22CD','df':map_df,'map_col':'LAD22CD'}
                        } #to format plotting below
    
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
    
    
    
    
    
    df_beef_geo=df_beef_clean[~df_beef_clean[geoplotting[plotting]['col']].isna()] #drop the rows with no lads code
    
    
    
    ##Spatial analysi for ###SOLID
    out_solid_region={} #{LAD:{manure_group:percenatge}}

    for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
        df_lad=df_beef_geo[df_beef_geo[geoplotting[plotting]['col']]==LAD]
        inner_dict={}
        df_lad=add_proprtion_of_items(df_lad, [b[0] for b in beef_items],prefix='beef')
        for m in solid_manure_items:
            df_for_calc=deepcopy(df_lad)
             
            if len(df_for_calc)==0: #if no data here
                weighted_mean=np.nan
            else:
                weighted_mean,weighted_std=calc_weighted_mean(df_for_calc,m+'_normed','beef_prop')
            
            n_farms=len(df_for_calc)
            n_cattle=df_for_calc.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
            inner_dict[m]=weighted_mean*100 #make it percent        
        out_solid_region[LAD]=inner_dict
        
    #join this data onto map_df    
    gdf=geoplotting[plotting]['df'] #pick it out for ease of syntax
    fig,axs=plt.subplots(nrows=int(np.ceil(len(solid_manure_items)/2)),ncols=2, figsize=(6, 9), constrained_layout=True)
    axs=axs.flatten()
    #I want to get a global vmax and min so extrac data first
    all_values=[]
    for m in solid_manure_items:
        all_values += [v[m] for v in out_solid_region.values()]
    
    vmin = min(all_values)
    vmax = max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'
    xmin, ymin, xmax, ymax = gdf.total_bounds #bounds of map
    
    for i,m in enumerate(solid_manure_items):
        all_values += [v[m] for v in out_solid_region.values()]
        mapper_dict={k:v[m] for k,v in out_solid_region.items()} #to join
        gdf[m+'_mean']=gdf[geoplotting[plotting]['map_col']].map(mapper_dict)
        #plot
        
        gdf.plot(column=m + '_mean', cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
        
    
        #axs[i].set_aspect('auto')
        axs[i].axis('off')
        wrapped_title = textwrap.fill(manure_storage_items_dict[m], width=30)
        axs[i].set_title(wrapped_title,fontsize=10)
    
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
    plt.savefig(save_dir+'solid_manure_{}.png'.format(geoplotting[plotting]['col']),dpi=300)      
    
    

    
    
    #Spatial analysis for liquid.
    out_liquid_region={} #{LAD:{manure_group:percenatge}}

    for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
        df_lad=df_beef_geo[df_beef_geo[geoplotting[plotting]['col']]==LAD]
        inner_dict={}
        df_lad=add_proprtion_of_items(df_lad, [b[0] for b in beef_items],prefix='beef')
        for m in liquid_manure_items:
            df_for_calc=deepcopy(df_lad)
             
            if len(df_for_calc)==0: #if no data here
                weighted_mean=np.nan
            else:
                weighted_mean,weighted_std=calc_weighted_mean(df_for_calc,m+'_normed','beef_prop')
            
            n_farms=len(df_for_calc)
            n_cattle=df_for_calc.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
            inner_dict[m]=weighted_mean*100 #make it percent        
        out_liquid_region[LAD]=inner_dict
        
    #join this data onto map_df    
    gdf=geoplotting[plotting]['df'] #pick it out for ease of syntax
    fig,axs=plt.subplots(nrows=int(np.ceil(len(liquid_manure_items)/2)),ncols=2, figsize=(10, 10), constrained_layout=True)
    axs=axs.flatten()
    #I want to get a global vmax and min so extrac data first
    all_values=[]
    for m in liquid_manure_items:
        all_values += [v[m] for v in out_liquid_region.values()]
    
    vmin = min(all_values)
    vmax = max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'
    xmin, ymin, xmax, ymax = gdf.total_bounds #bounds of map
    
    for i,m in enumerate(liquid_manure_items):
        all_values += [v[m] for v in out_liquid_region.values()]
        mapper_dict={k:v[m] for k,v in out_liquid_region.items()} #to join
        gdf[m+'_mean']=gdf[geoplotting[plotting]['map_col']].map(mapper_dict)
        #plot
        
        gdf.plot(column=m + '_mean', cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
        
    
        #axs[i].set_aspect('auto')
        axs[i].axis('off')
        wrapped_title = textwrap.fill(manure_storage_items_dict[m], width=35)
        axs[i].set_title(wrapped_title,fontsize=10)
    
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
    plt.savefig(save_dir+'liquid_manure_{}.png'.format(geoplotting[plotting]['col']),dpi=300)
    
    
    
   

        
        
  
    