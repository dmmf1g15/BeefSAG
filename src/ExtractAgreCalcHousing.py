'''
Code to extract housing % from agrcalc data. Here we use the AgreCalc beef types.
follows a similar approach to JAS analaysis

'Average number over 12 Months' column is the one that tells you how many units there are.

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

from ExtractJASManureHandlingData import map_df,nuts_df,itl_scot #for mapping

import textwrap

def add_prop_item(df,item):
    #To add in the poroprtion of the number of that item is in a given row over the full df
    #will make a new column with item_prop. The data frame returned will only have rows 
    #with that item type 
    reports=list(set(df['Report ID']))
    if type(item)!=list: # if item is not a list just do the one column:
        df_item = df.loc[df["Enterprise Sector Item"] == item].copy()
    
        # Keep only one row per Report ID (if duplicates, keep first)
        df_item = df_item.drop_duplicates(subset="Report ID")
    
        # Compute the denominator (total across all reports)
        total = df_item["Average number over 12 Months"].sum()
    
        # Add proportion column
        df_item[item + "_prop"] = df_item["Average number over 12 Months"] / total
        return df_item
    
    else: #its a list that need to be summed to get all animals in the list
        # Filter to only the items we care about
        df_items = df.loc[df["Enterprise Sector Item"].isin(item)].copy()
    
        # Deduplicate by Report ID per item
        df_items = df_items.drop_duplicates(subset=["Enterprise Sector Item", "Report ID"])
        total = df_items["Average number over 12 Months"].sum()
        # Compute proportions relative to this grand total
        df_items["prop"] = df_items["Average number over 12 Months"] / total
    
        return df_items             
 
            
def calc_weighted_mean(df,col,w_col):
    #to calculate weighted mean using two cols
    mean=df.apply(lambda row: row[col]*row[w_col],axis=1).sum()
    std=np.sqrt(df.apply(lambda row: (row[col]-mean)**2*row[w_col],axis=1).sum())
    return mean,std






agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20250728-SAGBeef_LAD.csv'


df_raw=pd.read_csv(agrecalc_path)
df_raw=df_raw.drop(columns=['Housed (%)'])
df_beef=df_raw[df_raw['Sector']=='Beef'] #only beef rows
beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)



years=list(set(df_beef['Year End'])) #unique beef animal types
years=[int(y) for y in years]
LADS= list(set(df_beef['LAD_CODE'])) #unique local authroury district codes 
NUTS2=list(set(df_beef['NUTS2']))
NUTS2=[n for n in NUTS2 if n in itl_scot] #filter down to scotland.

manure_cols=['Pasture (%)', 'Hill ground (%)', 'Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)','Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']
housed_manure_cols=['Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)', 'Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']






if __name__ == '__main__':
    save_dir='C:\\Users\\dfletcher\\Documents\\BeefSAG\\output\\Housing\\'
    beef_farm_types=list(set(df_beef['Enterprise Item'])) #types of beef farm, might be useful later.
    df_beef['Housed (%)']=df_beef[housed_manure_cols].sum(axis=1) #summing all manure to get housed (%)
    
    housing_cols=['Housed (%)','Pasture (%)', 'Hill ground (%)']
    
    plotting='nuts' #chose #nuts or LAD to decide what gets plot.
    
    geoplotting={'nuts':{'shapes':NUTS2,'col':'NUTS2','df':nuts_df,'map_col':'ITL225CD'},
                 'LAD':{'shapes':LADS,'col':'LAD_CODE','df':map_df,'map_col':'LAD22CD'}
                        } #to format plotting below
    
    farm_years1=len(list(set(df_beef['Report ID'])))
    print('Started with {} many farm years'.format(farm_years1))
    
    
    #drop rows who's housed + pature +hill isnt 100
    rows1=len(df_beef)
    print('Started with {} rows'.format(rows1))
    df_beef_clean=df_beef[df_beef[manure_cols].sum(axis=1)==100]
    rows2=len(df_beef_clean)
    farm_years2=len(list(set(df_beef_clean['Report ID'])))
    print('Dropped {} rows since thier manure percentage didnt sum to 100'.format(rows1-rows2))
    print('Dropped {} farm years since thier manure percentage didnt sum to 100'.format(farm_years1-farm_years2))
    
    
    '''
    #Calculate mean and std for all farms and years and make pdfs.
    for_pdf='Housed (%)' #choose something from hosuing_cols
    # we loop over beef items and extract means etc over all years
    out_dict={}#{bt:{mean_housing1:x,std_housing1,....,n_farm_years,n_cattle_years}
    fig,axs=plt.subplots(nrows=int(np.ceil(len(beef_types)/2)),ncols=2,figsize=(1 * len(beef_types), 12))
    axs=axs.flatten()
    for i,bt in enumerate(beef_types):
        inner_dict={}
        df_bt=add_prop_item(df_beef_clean,bt) #this shortens the data down to the bt rows.
        df_bt=df_bt[~df_bt[bt+'_prop'].isna()]
        n_farm_years=len(list(set(df_bt['Report ID'])))
        n_cattle_years=df_bt['Average number over 12 Months'].sum()
        inner_dict['n_farm_years']=n_farm_years
        inner_dict['n_cattle_years']=n_cattle_years
        
        for h in housing_cols:
            mean,std=calc_weighted_mean(df_bt,h,bt+'_prop')
            inner_dict[h+'_mean']=mean
            inner_dict[h+'_std']=std
        out_dict[bt]=inner_dict
        axs[i].hist(df_bt[for_pdf],weights=df_bt[bt+'_prop'],density=True,bins=100)
        axs[i].set_xlabel(for_pdf,fontsize=9)
        axs[i].set_ylabel('Probability',fontsize=9)
        axs[i].tick_params(axis='both', labelsize=9)
        wrapped_title = textwrap.fill(bt+', n-farm-years={}'.format(n_farm_years), width=70)
        axs[i].set_title(wrapped_title,fontsize=9)
    df_out=pd.DataFrame(out_dict)
    df_out=df_out[sorted(df_out.columns)] # arrange alphabeticaly
    df_out.to_csv(save_dir+'housing_means_AC.csv')
    if len(beef_types) %2 ==1: #its odd number of axs
        axs[-1].axis('off')     #had not needed AXIS  
    plt.tight_layout()    
    plt.savefig(save_dir+for_pdf+'_hist.png',dpi=300)
    plt.close('all')
    '''
    
    
    
    
    '''
    #In time plots
    fig,axs=plt.subplots(nrows=5,ncols=int(np.ceil(len(beef_types)/5)), figsize=(9, 10), constrained_layout=True)
    axs=axs.flatten() #an axs for each beef type
    
    housing_colors=['#1f77b4','#ff7f0e','#2ca02c']
    
    
    for i,bt in enumerate(beef_types):
        inner_dict={h:{'mean':[],'std':[]} for h in housing_cols} #{h:{mean:[],std=[],n_cattle=[]}
        n_cattle=[]
        n_farms=[]
        for y in years: 
            df_beef_year=df_beef_clean[df_beef_clean['Year End']==y] #filter down to years
            df_bt=add_prop_item(df_beef_year,bt) #this shortens the data down to the bt rows.
            df_bt=df_bt[~df_bt[bt+'_prop'].isna()]
            n_farms.append(len(list(set(df_bt['Report ID']))))
            n_cattle.append(df_bt['Average number over 12 Months'].sum())
            for h in housing_cols: #loop through to get wieghted means
                mean,std=calc_weighted_mean(df_bt,h,bt+'_prop')
                inner_dict[h]['mean'].append(mean) #save it
                inner_dict[h]['std'].append(std)
        #now make plots
        for j,h in enumerate(housing_cols):
            axs[i].errorbar(years,inner_dict[h]['mean'],yerr=inner_dict[h]['std'],color=housing_colors[j],label=h,capsize=6)
            axs[i].set_ylabel('%')
            axs[i].tick_params(axis='x',labelsize=9)
            axs[i].set_xticks([int(years[0]), int(years[-1])])
        
        axs[i].set_title(bt)
        #makke legend
        #if i==0:
            #[i].legend()
    legend_ax = fig.add_axes([0.9, 0.1, 0.1, 0.8])  # [left, bottom, width, height]
    legend_ax.axis('off')  # hide the axes
    fig.legend(
        handles=[plt.Line2D([0], [0], color=housing_colors[i], label=h) for i, h in enumerate(housing_cols)],
        loc='upper center',   # position on the figure
        ncol=len(housing_cols),#len(housing_cols),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04))
    #fig.subplots_adjust(bottom=0.01)
    fig.savefig(save_dir+'hosuing_means_time_AC2.png',bbox_inches='tight',dpi=300) 
    '''
    
    
    #Spatial plots. Will only do housing % 
    
    ##First extract means for each region and beef type
    df_beef_geo=df_beef_clean[~df_beef_clean[geoplotting[plotting]['col']].isna()] #drop the rows with no lads code
    rows3=len(df_beef_geo)
    farm_years3=len(list(set(df_beef_geo['Report ID'])))
    print('Dropped {} farm-years since there was no valid postcode'.format(farm_years2-farm_years3))
    out_housed_region={} #{LAD:{bt:{housed:housed...,n_cattle:}}}
    for counter,LAD in enumerate(geoplotting[plotting]['shapes']):
        df_lad=df_beef_geo[df_beef_geo[geoplotting[plotting]['col']]==LAD]
        inner_dict={}
        for bt in beef_types:
            df_lad_bt=add_prop_item(df_lad,bt)# add in proportions. Shortens df down to only the bt rows
            n_farms = len(list(set(df_lad_bt['Report ID'])))
            n_cattle = df_lad_bt['Average number over 12 Months'].sum()
            mean,std=calc_weighted_mean(df_lad_bt,'Housed (%)',bt+'_prop')
            d={'n_farms':n_farms,'n_cattle':n_cattle,'Housed (%)_mean':mean,'Housed (%)_std':std}
            inner_dict[bt]=d
        out_housed_region[LAD]=inner_dict
    
    #Join dcitionary onto df and plot
    key_to_plot='Housed (%)_mean'
    fig,axs=plt.subplots(nrows=4,ncols=3, figsize=(7, 10), constrained_layout=True)
    axs=axs.flatten() #an axs for each beef type
    cmap = 'viridis'
    
    #To normalise color map 0-100
    norm = Normalize(vmin=0, vmax=100)
    
    gdf=geoplotting[plotting]['df'] #pick it out for ease of syntax
    for i,bt in enumerate(beef_types):
        mapper_dict={k:v[bt][key_to_plot] for k,v in out_housed_region.items()} #to join
        
  
        gdf[bt+'_mean']=gdf[geoplotting[plotting]['map_col']].map(mapper_dict)
        #gdf=gdf.dropna(subset=[bt+'_mean'])
        gdf_plot=gdf.plot(column=bt + '_mean', cmap=cmap,ax=axs[i],norm=None, legend=True,missing_kwds={'color': 'lightgrey'})
        #axs[i].set_aspect('auto')
        axs[i].axis('off')
        wrapped_title = textwrap.fill(bt, width=40)
        axs[i].set_title(wrapped_title,fontsize=14)
        colorbar = gdf_plot.get_figure().get_axes()[1]  # second axis is the colorbar
        colorbar.set_ylabel('Hosuing %', fontsize=12)
        
    #plt.show()
    axs[-1].axis('off')
    axs[-2].axis('off')
    fig.savefig(save_dir+key_to_plot+'{}_spatial.png'.format(geoplotting[plotting]['col']),dpi=300)
    
    
        
    
    
    