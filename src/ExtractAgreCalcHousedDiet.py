'''
Code to process AgreCalc data and pick out diet of animal types

'Enterprise Sector Item' column has beef or crop types in it

'Fed or Used for Bedding (t)'

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

from ExtractJASManureHandlingData import map_df,nuts_df #for mapping
from ExtractAgreCalcHousing import add_prop_item, calc_weighted_mean,itl_scot


import textwrap

save_dir='../output/ManureHandlingSystem/AC/'
agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20250728-SAGBeef_LAD.csv'




df_raw=pd.read_csv(agrecalc_path)
df_raw=df_raw.drop(columns=['Housed (%)'])
df_beef=df_raw[df_raw['Sector']=='Beef'] #only beef rows
beef_types=list(set(df_beef['Enterprise Sector Item'])) #unique beef animal types
beef_types=sorted(beef_types)

years=list(set(df_beef['Year End'])) #unique years
LADS= list(set(df_beef['LAD_CODE'])) #unique local authroury district codes
NUTS2=list(set(df_beef['NUTS2']))
NUTS2=[n for n in NUTS2 if n in itl_scot] #filter down to scotland.


#Enterprise sector items for rows to do with diet are those which have >0 inb column 'Fed or Used for Bedding (t)'
df_raw


