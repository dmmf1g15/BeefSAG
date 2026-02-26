'''
Data and functions used by all scripts
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd

def parish_to_LAD22CODE(parish_no,converter_df):
    row=converter_df[converter_df['parish']==parish_no]
    #print(row['LAcode'])
    if not row.empty:
        return row['LAcode'].iloc[0]
    else:
        return np.nan
    
def calc_weighted_mean(df,col,w_col):
    #to calculate weighted mean using two cols
    tot=df[w_col].sum()
    mean=df.apply(lambda row: row[col]*row[w_col]/tot,axis=1).sum()
    std=np.sqrt(df.apply(lambda row: (row[col]-mean)**2*row[w_col]/tot,axis=1).sum())
    return mean,std


#Code for maps!
JAC_path='D:\\JAS\\jas2holdings2023\\jas2holdings2023.dta'
nuts_path='D:\\SOURCED-DATA\\NUTS\\ITL2_JAN_2025_UK_BFC\\ITL2_JAN_2025_UK_BFC.shp'
map_path='D:\\SOURCED-DATA\\Admin-Regions\\Local_Authority_Districts_(December_2022)_Boundaries_UK_BFC\\Local_Authority_Districts_(December_2022)_Boundaries_UK_BFC\\LAD_DEC_2022_UK_BFC.shp'
map_df=gpd.read_file(map_path)
itl_scot=['TLM0','TLM1','TLM2','TLM3', 'TLM5', 'TLM9'] #list of itl_codes level 2 2025 that are in scotland.
nuts_df=gpd.read_file(nuts_path) #not really nuts but itl2
nuts_df=nuts_df[nuts_df['ITL225CD'].isin(itl_scot)] #filter to scotland

converter_path='D:\\SOURCED-DATA\\NUTS\\ParishGeographyLookups.xlsx'
converter_df=pd.read_excel(converter_path)
df=pd.read_stata(JAC_path)
df['LAD22CD']=df.apply(lambda row: parish_to_LAD22CODE(row['parish'],converter_df),axis=1) #add in a new column for mapping
LADS=list(set(df['LAD22CD']))
map_df=map_df[map_df['LAD22CD'].isin(LADS)] #filter to scotlandd

###WORK FROM HERE. USE BELOW TO CONVERT JAC to ITL2. Note use the LAD output above as input LAD -> ITLS map in the cav below.
nuts_converter_path='D:\\SOURCED-DATA\\NUTS\\LAD_(December_2024)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2025)_Lookup_in_the_UK.csv'
converter_nuts_df=pd.read_csv(nuts_converter_path)
#make a dictionary to convert
nuts_converter_dict=dict(zip(converter_nuts_df['LAD24CD'],converter_nuts_df['ITL225CD']))
#LAD24CD -> ITL225 mapper.
nuts_converter_dict[np.nan] = np.nan #make sure nan gets mapped to nan
df['ITL225CD']=df.apply(lambda row: nuts_converter_dict[row['LAD22CD']],axis=1)
NUTS2=list(set(df['ITL225CD']))