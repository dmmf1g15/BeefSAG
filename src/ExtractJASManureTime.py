# -*- coding: utf-8 -*-
'''
To work out  manure holding times
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
from ExtractJASManureHandlingData import  beef_items, manure_time_items, df,map_df, converter_df, add_proprtion_of_items,LADS
import textwrap


if __name__ == "__main__":
    #df is the JAC2022 where we have mapped the farms to a Local authroity for mapping.
    save_dir='../output/ManureStorageTime/'
    
    beef_farm_cutoff=10# how many beef cows required to be called a beef farm
    df_beef_cols=df[[b[0] for b in beef_items]] #only beef item cols
    df_beef=df[df_beef_cols.sum(axis=1)>beef_farm_cutoff] #only include farms which have more than x  beef cows in total.
    print("{} out of {} farms in scotland are beef farms with this rule".format(len(df_beef),len(df)))  
    
    df_beef_clean=df_beef[~df_beef[[m[0] for m in manure_time_items]].isna().all(axis=1)] #they haven;t filled it in
    print("Removed {} farms since they didn't fill in manure storage time sectiion".format(len(df_beef)-len(df_beef_clean)))
    filt1len=len(df_beef_clean)
    
    df_beef_clean=df_beef[df_beef[[m[0] for m in manure_time_items]].le(36).all(axis=1)]
    print("Removed {} farms since they sayed they stored some manure for more than 36 months".format(filt1len-len(df_beef_clean)))
    
    total_farms=len(df_beef_clean)
    total_animals=df_beef_clean[[b[0] for b in beef_items]].sum().sum() #total animals considered
    out={} #to save means and stds for each type in manure_time_items
    fig,axs=plt.subplots(int(np.ceil(len(manure_time_items)/2)),2)
    axs=axs.flatten()
    i=0
    for m in manure_time_items: #m is a tuple for each time type.
        df_for_calc=df_beef_clean[~df_beef_clean[m[0]].isna()] #remove nans in this manure type
        df_for_calc=df_for_calc[df_for_calc[m[0]]>0]#if the answe is 0 don't trust it! Only keep the cols we need#
        
        df_for_calc_prop=add_proprtion_of_items(df_for_calc, [b[0] for b in beef_items],prefix='beef') #add in the proportion of all beef animals after we have removed the 0s
        
        
        weighted_mean=df_for_calc_prop.apply(lambda row: row[m[0]]*row['beef'+'_prop'],axis=1).sum()
        weighted_std=np.sqrt(df_for_calc_prop.apply(lambda row: (row[m[0]]-weighted_mean)**2*row['beef'+'_prop'],axis=1).sum())
        n_farms=len(df_for_calc_prop)
        #calcualte how many cows included in calculation
        n_cattle=df_for_calc_prop.apply(lambda row: row[[b[0] for b in beef_items]].sum(),axis=1).sum()
        out[m[1]]={'mean':weighted_mean,'std':weighted_std,'n-farm':n_farms,'farm_percent':n_farms/total_farms*100,'n-cattle':n_cattle,'cattle_percent':n_cattle/total_animals*100} #save out the mean,std
        #histogram
        axs[i].hist(df_for_calc_prop[m[0]],weights=df_for_calc_prop['beef_prop'],density=True,bins=50)
        axs[i].set_xlabel('Storage time (months))',fontsize=7)
        axs[i].set_ylabel('Probability',fontsize=7)
        axs[i].tick_params(axis='both', labelsize=7)
        wrapped_title = textwrap.fill(m[1]+', n-holdings={}'.format(n_farms), width=70)
        axs[i].set_title(wrapped_title,fontsize=5)
        i+=1
        
        #s1=m[1].replace('/', ' or ')
    axs[-1].axis('off')     #had not needed AXIS  
    plt.tight_layout()    
    plt.savefig(save_dir+'manure_time_hist.png',dpi=300)
    plt.close('all')
    #save csv:
    df_out=pd.DataFrame.from_dict(out)
    df_out.to_csv(save_dir+'manure_time_means.csv')
    '''
    #Do regional plots
    out_region={} #{LAD:{manure_group:time}}
    counter=0
    for LAD in LADS:
        print("on {} out of {}".format(counter,len(LADS)))
        df_lad=df_beef_clean[df_beef_clean['LAD22CD']==LAD] #this particular LAD
        inner_dict={}
        for m in manure_time_items: #m is a tuple
            df_for_calc=df_lad[~df_lad[m[0]].isna()] #remove nans in this manure type
            df_for_calc=df_for_calc[df_for_calc[m[0]]>0]#if the answe is 0 don't trust it! Only keep the cols we need#
            df_for_calc_prop=add_proprtion_of_items(df_for_calc, [b[0] for b in beef_items],prefix='beef') 
            if len(df_for_calc_prop)==0: #if no data here
                weighted_mean=np.nan
            else:
                weighted_mean=df_for_calc_prop.apply(lambda row: row[m[0]]*row['beef'+'_prop'],axis=1).sum()
            inner_dict[m[1]]=weighted_mean
        out_region[LAD]=inner_dict
        counter+=1
    #join this data onto map_df    
    out={} #to save means and stds for each type in manure_time_items
    fig,axs=plt.subplots(ncols=int(np.ceil(len(manure_time_items)/2)),nrows=2, figsize=(9,1 * len(manure_time_items)), constrained_layout=True)
    axs=axs.flatten()
    #I want to get a global vmax and min so extrac data first
    all_values=[]
    for m in manure_time_items:
        all_values += [v[m[1]] for v in out_region.values()]
    
    vmin = min(all_values)
    vmax = max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'

    
    
    xmin, ymin, xmax, ymax = map_df.total_bounds #bounds of map
    
    for i,m in enumerate(manure_time_items):
        all_values += [v[m[1]] for v in out_region.values()]
        mapper_dict={k:v[m[1]] for k,v in out_region.items()} #to join
        map_df[m[0]+'_mean']=map_df['LAD22CD'].map(mapper_dict)
        #plot
        
        
        map_df.plot(column=map_df[m[0] + '_mean'], cmap=cmap, norm=norm,ax=axs[i], legend=False,missing_kwds={'color': 'lightgrey'})
        

        #axs[i].set_aspect('auto')
        axs[i].axis('off')
        wrapped_title = textwrap.fill(m[1], width=40)
        axs[i].set_title(wrapped_title,fontsize=7)

    axs[-1].axis('off')     #had not needed AXIS                
    # Shared colorbar
    #fig.subplots_adjust(right=0.8) #space for colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label('Average Storage time (months)', fontsize=10)
    
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0, right=0.9)  # wspace controls space between plots
    plt.savefig(save_dir+'manure_time_regional.png',dpi=300)
    '''
                   
                