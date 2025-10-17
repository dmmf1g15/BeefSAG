'''
To add in a new column to the agrecalc data which has the LAD code based on post code.
'''
import pandas as pd
import requests
import numpy as np
def post_code_to_LAD(post_code):
    url = "https://api.postcodes.io/postcodes"

    payload = {"postcodes":[post_code]}
    headers = {
      'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, json=payload)
    d=response.json()
    if response.status_code ==200 and (d['result'][0]['result'] is not None): #last condition handles when None is resutrend
        LAD_code=d['result'][0]['result']['codes']['admin_district']
        nuts3=d['result'][0]['result']['codes']['nuts'] #this is actually nuts3
        nuts2=nuts3[0:4] #slice code to nuts2.
        return LAD_code,nuts2,response.status_code
    else: #return nan 
        return np.nan,np.nan,response.status_code



agrecalc_path='D:\\AgreCalc\\BeefSAG_2025\\20251014-SAGBeef.csv'
out_path='D:\\AgreCalc\\BeefSAG_2025\\20251014-SAGBeef_LAD.csv'



df=pd.read_csv(agrecalc_path)


#first get uique postcodes and make a dictionary to apply

PCs= list(set(df['Post Code']))

pc_lad_dict={}
pc_nuts2_dict={}

for i,pc in enumerate(PCs):
    print('on postcode {} of {}'.format(i+1, len(PCs)))
    
    LAD,nuts2, status = post_code_to_LAD(pc)
    if status==200: #good response
        pc_nuts2_dict[pc]=nuts2
        pc_lad_dict[pc]=LAD
    else:
        pc_nuts2_dict[pc]=np.nan
        pc_lad_dict[pc]=np.nan
        print('Bad status for nuts2, {}'.format(status))
    
        

df['LAD_CODE']=df['Post Code'].map(pc_lad_dict)

df['NUTS2']=df['Post Code'].map(pc_nuts2_dict)

df.to_csv(out_path)
