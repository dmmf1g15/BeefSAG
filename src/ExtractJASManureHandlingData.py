'''
Code to process JAS2023 and pick out manure management percentages of solid vs liquid for beef
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import textwrap

def add_proprtion_of_items(df,list_of_cols,prefix=''):
    #will work out what proportion of col that row has and 
    #adds it as a new column. Will also do sum of items
    total=0
    for item in list_of_cols:
        t=df[item].sum()
        total=total+t
        df[item+'{}_prop'.format(prefix)]=df.apply(lambda row: row[item]/t,axis=1)
    df[prefix+'_prop']=df.apply(lambda row: np.nansum(row[list_of_cols])/total,axis=1)
    return df


def parish_to_LAD22CODE(parish_no,converter_df):
    row=converter_df[converter_df['parish']==parish_no]
    #print(row['LAcode'])
    if not row.empty:
        return row['LAcode'].iloc[0]
    else:
        return np.nan

def calc_weighted_mean(df,col,w_col):
    #to calculate weighted mean using two cols
    mean=df.apply(lambda row: row[col]*row[w_col],axis=1).sum()
    std=np.sqrt(df.apply(lambda row: (row[col]-mean)**2*row[w_col],axis=1).sum())
    return mean,std
   

beef_items=[('cts301','Female beef cattle under 1 yr'),('cts303','Female beef cattle aged 1-2 yrs'),('cts307','Female beef cattle without offspring aged 2 yrs and over'),
            ('cts311','Male cattle aged 2 yrs and over'),('cts310','Male cattle aged 1-2 yrs'),('cts309','Male cattle under 1 yr'),
            ('cts305','Female beef cattle with offspring aged 2 yrs and over')]

beef_groups={'<1 Yr':['cts301','cts309'],
             '>1 Yr':['cts303','cts307','cts311','cts310','cts305']}


manure_storage_items=[('item5115', 'Manure solid storage in heaps %'),
                        ('item5116', 'Manure stored in compost piles %'),
                        ('item5117', 'Manure stored in pits below animal confinement %'),
                        ('item5118', 'Manure stored in deep litter systems %'),
                        ('item5119', 'Liquid manure/slurry storage without cover %'),
                        ('item5120', 'Liquid manure/slurry storage with permeable cover'),
                        ('item5121', 'Liquid manure/slurry storage with impermeable cover %'),
                        ('item5122', 'Manure stored in other facilities (not elsewhere classified) %'),
                        ('item5123', 'Daily spread %'),
                        ('item5132', 'Other %')]



solid_manure_items=['item5115','item5116','item5117','item5118','item5122'] #items that are solid manure
liquid_manure_items=['item5119','item5120','item5121','item5123'] #items that are liquid manure.
    

manure_time_items=[('item5124', 'Manure stored in compost piles (months)'),
                    ('item5125', 'Manure stored in pits below animal confinement (months)'),
                    ('item5126', 'Manure stored in deep litter systems (months)'),
                    ('item5127', 'Liquid manure/slurry storage (months)'),
                    ('item5128', 'Manure stored in other facilities (not elsewhere classified) (months)')
                        ]

JAC_path='D:\\JAS\\jas2holdings2023\\jas2holdings2023.dta'
nuts_path=r'D:\SOURCED-DATA\NUTS\ITL2_JAN_2025_UK_BFC\ITL2_JAN_2025_UK_BFC.shp'
map_path='D:\\SOURCED-DATA\\Admin-Regions\\Local_Authority_Districts_(December_2022)_Boundaries_UK_BFC\\Local_Authority_Districts_(December_2022)_Boundaries_UK_BFC\\LAD_DEC_2022_UK_BFC.shp'
map_df=gpd.read_file(map_path)
itl_scot=['TLM0','TLM1','TLM2','TLM3', 'TLM5', 'TLM9'] #list of itl_codes level 2 2025 that are in scotland.
nuts_df=gpd.read_file(nuts_path) #not really nuts but itl2
nuts_df=nuts_df[nuts_df['ITL225CD'].isin(itl_scot)] #filter to scotland



converter_path=r'D:\SOURCED-DATA\NUTS\ParishGeographyLookups.xlsx'
converter_df=pd.read_excel(converter_path)
df=pd.read_stata(JAC_path)
df['LAD22CD']=df.apply(lambda row: parish_to_LAD22CODE(row['parish'],converter_df),axis=1) #add in a new column for mapping
LADS=list(set(df['LAD22CD']))
map_df=map_df[map_df['LAD22CD'].isin(LADS)] #filter to scotlandd

###WORK FROM HERE. USE BELOW TO CONVERT JAC to ITL2. Note use the LAD output above as input LAD -> ITLS map in the cav below.
nuts_converter_path=r'D:\SOURCED-DATA\NUTS\LAD_(December_2024)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2025)_Lookup_in_the_UK.csv'
converter_nuts_df=pd.read_csv(nuts_converter_path)
#make a dictionary to convert
nuts_converter_dict=dict(zip(converter_nuts_df['LAD24CD'],converter_nuts_df['ITL225CD']))
#LAD24CD -> ITL225 mapper.
nuts_converter_dict[np.nan] = np.nan #make sure nan gets mapped to nan
df['ITL225CD']=df.apply(lambda row: nuts_converter_dict[row['LAD22CD']],axis=1)
NUTS2=list(set(df['ITL225CD']))

if __name__=="__main__":
    beef_farm_cutoff=10# how many beef cows required to be called a beef farm
    save_dir='../output/ManureHandlingSystem/'
    
    
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
    filt2len=len(df_beef_clean)
    print("Removed {} farms since they reported more than 100% manure storage percentage".format(filt1len-filt2len))
    

    #First lets do solid manure vs liquid manure for <1 Yr and >1 Yr
    
    #remove rows that don't have any solid or lqiued
    filt2len=len(df_beef_clean)
    df_beef_clean=df_beef_clean[df_beef_clean[solid_manure_items+liquid_manure_items].sum(axis=1)>0]
    print("Removed {} farms since they had a total of 0 liquid and solid manure".format(filt2len-len(df_beef_clean)))
    
    
    
    df_solid_liquid=df_beef_clean.copy(deep=True)
    df_solid_liquid['solid_percent']=df_solid_liquid.apply(lambda row: np.nansum(row[solid_manure_items])/(np.nansum(row[solid_manure_items])+np.nansum(row[liquid_manure_items])),axis=1) #to make sure it adds to 100%
    df_solid_liquid['liquid_percent']=df_solid_liquid.apply(lambda row: np.nansum(row[liquid_manure_items])/(np.nansum(row[solid_manure_items])+np.nansum(row[liquid_manure_items])),axis=1)
    
    '''
    #Caluclate for whole country
    liquid_percents={}
    liquid_sds={}
    solid_percents={}
    solid_sds={}
    
    
    
    fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(10, 10))
    axs=axs.flatten()
    i=0
    for k,v in beef_groups.items():
        #add in proportion of beef cows owned so we can do weighted averages
        df_solid_liquid=add_proprtion_of_items(df_solid_liquid, v,prefix=k)
        
        
        solid_tot=df_solid_liquid.apply(lambda row: row['solid_percent']*row[k+'_prop'],axis=1).sum() #weighted and makes sure the liqud+solid =100% in the farm
        liquid_tot=df_solid_liquid.apply(lambda row: row['liquid_percent']*row[k+'_prop'],axis=1).sum() #weighted average
        liquid_percents[k]=liquid_tot
        solid_percents[k]=solid_tot
        liquid_sd=np.sqrt(df_solid_liquid.apply(lambda row: (row['liquid_percent']-liquid_tot)**2*row[k+'_prop'],axis=1).sum())
        liquid_sds[k]=liquid_sd
        
        solid_sd=np.sqrt(df_solid_liquid.apply(lambda row: (row['solid_percent']-solid_tot)**2*row[k+'_prop'],axis=1).sum())
        solid_sds[k]=solid_sd
        
        
        n_farm_years=len(df_solid_liquid)
        
        axs[i].hist(df_solid_liquid['liquid_percent'],weights=df_solid_liquid[k+'_prop'],bins=100,density=True)
        axs[i].set_xlabel('Proportion of Manure',fontsize=9)
        axs[i].set_ylabel('Probability',fontsize=9)
        axs[i].tick_params(axis='both', labelsize=9)
        wrapped_title = textwrap.fill(k+' liquid_percent'+', n-enterprise={}'.format(n_farm_years), width=70)
        axs[i].set_title(wrapped_title,fontsize=9)
    
        axs[i+1].hist(df_solid_liquid['solid_percent'],weights=df_solid_liquid[k+'_prop'],bins=100,density=True)
        axs[i+1].set_xlabel('Proportion of Manure',fontsize=9)
        axs[i+1].set_ylabel('Probability',fontsize=9)
        axs[i+1].tick_params(axis='both', labelsize=9)
        wrapped_title = textwrap.fill(k+' solid_percent'+', n-enterprise={}'.format(n_farm_years), width=70)
        axs[i+1].set_title(wrapped_title,fontsize=9)
        
        i=i+2
        
    
        
    plt.savefig(save_dir+ 'histogram.png')    
    '''
    
    
    df_beef_geo=df_solid_liquid[~df_solid_liquid[geoplotting[plotting]['col']].isna()] #drop the rows with no lads code
    filt3len=len(df_beef_geo)
    print('Dropped {} farm since there was no valid postcode'.format(filt2len-filt3len))
    out_liquid_region={} #{LAD:{<1:{liquid_mean:x,liquid_std}}}
    for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
        df_lad=df_beef_geo[df_beef_geo[geoplotting[plotting]['col']]==LAD]
        inner_dict={}
        for k,v in  beef_groups.items():
            df_lad_bt=add_proprtion_of_items(df_lad, v,prefix=k)
            liquid_mean,liquid_std=calc_weighted_mean(df_lad_bt,'liquid_percent',k+'_prop')
            solid_mean,solid_std=calc_weighted_mean(df_lad_bt,'solid_percent',k+'_prop')

            n_farms=len(df_lad_bt)
            n_cattle=df_lad_bt[v].sum().sum()
            d={'n_farms':n_farms,'n_cattle':n_cattle,'liquid_mean':liquid_mean,'liquid_std':liquid_std,
               'solid_mean':solid_mean,'solid_std':solid_std}
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
    
    
    
    
    
    
    
    
    
    
