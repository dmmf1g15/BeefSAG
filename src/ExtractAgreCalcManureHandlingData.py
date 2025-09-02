'''
Code to process JAC2023 and pick out manure management percentages of solid vs liquid for beef

IN JAC anlaysis Solid was 'Manure solid storage in heaps %','Manure stored in compost piles %', 'Manure stored in pits below animal confinement %', 'Manure stored in deep litter systems,
'Manure stored in deep litter systems %', 'Manure stored in other facilities (not elsewhere classified) %'

And liquid: 'Liquid manure/slurry storage without cover %', 'Liquid manure/slurry storage without cover %', 'Liquid manure/slurry storage with impermeable cover %', 'Daily spread %'

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

from ExtractJASManureHandlingData import map_df #for mapping
#also nuts
nuts_path=r'D:\SOURCED-DATA\NUTS\NUTS_RG_20M_2024_3035.shp'
nuts_df=gpd.read_file(nuts_path)

import textwrap


agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20250728-SAGBeef_LAD.csv'

df_raw=pd.read_csv(agrecalc_path)
df_raw=df_raw.drop(columns=['Housed (%)'])
df_beef=df_raw[df_raw['Sector']=='Beef'] #only beef rows
beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)



years=list(set(df_beef['Year End'])) #unique beef animal types
LADS= list(set(df_beef['LAD_CODE'])) #unique local authroury district codes

housed_manure_cols=['Liquid Slurry (%)', 'Solid storage (FYM) (%)','Pit storage (Slats) (%)', 'Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']

liquid_cols=['Liquid Slurry (%)','Pit storage (Slats) (%)']

solid_cols=['Solid storage (FYM) (%)','Deep bedding (retained > 1yr) (%)', 'Anaerobic digestion (%)']