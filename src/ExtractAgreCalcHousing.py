'''
Code to extract housing % from agrcalc data. Here we use the AgreCalc beef types.
follows a similar approach to JAS analaysis

'Average number over 12 Months' column is the one that tells you how many units there are.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def add_prop_item(df,item):
    #To add in the poroprtion of the number of that item is in a given row over the full df
    #will make a new column with item_prop. The data frame returned will only have rows 
    #with that item type 
          
    df_item=df[df['Enterprise Sector Item']==item] #crop down to right items.
    total_items=df_item['Average number over 12 Months'].sum(axis=0)
    df_item[item+'_prop']=df_item['Average number over 12 Months']/total_items
    return df_item


def calc_weighted_mean(df,col,w_col):
    #to calculate weighted mean using two cols
    mean=df.apply(lambda row: row[col]*row[w_col],axis=1).sum()
    std=np.sqrt(df.apply(lambda row: (row[col]-mean)**2*row[w_col],axis=1).sum())
    return mean,std



agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20250728-SAGBeef.csv'


df_raw=pd.read_csv(agrecalc_path)
df_raw=df_raw.drop(columns=['Housed (%)'])
df_beef=df_raw[df_raw['Sector']=='Beef'] #only beef rows
beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)



years=list(set(df_beef['Year End'])) #unique beef animal types

manure_cols=['Pasture (%)', 'Hill ground (%)', 'Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)','Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']
housed_manure_cols=['Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)', 'Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']



if __name__ == '__main__':
    save_dir='C:\\Users\\dfletcher\\Documents\\BeefSAG\\output\\Housing\\'
    beef_farm_types=list(set(df_beef['Enterprise Item'])) #types of beef farm, might be useful later.
    df_beef['Housed (%)']=df_beef[housed_manure_cols].sum(axis=1) #summing all manure to get housed (%)
    
    housing_cols=['Housed (%)','Pasture (%)', 'Hill ground (%)']
    
    farm_years1=len(list(set(df_beef['Report ID'])))
    print('Started with {} many farm years'.format(farm_years1))
    
    
    #drop rows who's housed + pature +hill isnt 100
    rows1=len(df_beef)
    print('Started with {} rows'.format(rows1))
    df_beef_clean=df_beef[df_beef[manure_cols].sum(axis=1)==100]
    rows2=len(df_beef)
    farm_years2=len(list(set(df_beef['Report ID'])))
    print('Dropped {} rows since thier manure percentage didnt sum to 100'.format(rows1-rows2))
    print('Dropped {} farm years since thier manure percentage didnt sum to 100'.format(farm_years1-farm_years2))
    
    
    '''
    #Calculate mean and std for all farms and years
    # we loop over beef items and extract means etc over all years
    out_dict={}#{bt:{mean_housing1:x,std_housing1,....,n_farm_years,n_cattle_years}
    for bt in beef_types:
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
    
    df_out=pd.DataFrame(out_dict)
    df_out=df_out[sorted(df_out.columns)] # arrange alphabeticaly
    df_out.to_csv(save_dir+'housing_means_AC.csv')
    '''
    #Now we do it over years to make plots
    fig,axs=plt.subplots(nrows=int(np.ceil(len(beef_types)/2)),ncols=2, figsize=(1 * len(beef_types), 9), constrained_layout=True)
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
        
        axs[i].set_title(bt)
        #makke legend
        if i==0:
            axs[i].legend()
       
    fig.savefig(save_dir+'hosuing_means_time_AC.png',dpi=300) 
        
    
    
    