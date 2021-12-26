# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:46:23 2021

@author: nkushwaha
"""

import io
import os
import re
#import argparse
import pandas as pd
from pathlib import Path
from kitts.config import ANNOTATED_DIR, EXCLUDE, MAPQKEY, MAPQURL
from kitts.utils import dataset_utils


def accumulate_dataframe(filepath = ANNOTATED_DIR):
    """Accumulates CSV file in mentoined path to a composit Dataframe"""    
     #Checking existence of files for given hastag and calling collect_post funtion
    if os.path.isdir(filepath):
        print(f'Source Path exists: {filepath}')
        df_list = []
        counter = 0
        try:            
            for file in Path(filepath).glob('*.csv'):            
                df = pd.read_csv(file, index_col=None, header=0)
                df_list.append(df)
                counter += 1
            Bigframe = pd.concat(df_list, axis=0, ignore_index=True)
            print(f'Combined {counter} files into one dataframe')
            return dedup_dataframe(Bigframe)
        except Exception as e:
            print(e)
    else:        
        print(f'Source Path does not exists: {filepath}')
        
def dedup_dataframe(dataframe):
    """Removes duplicate data based on shortcode such that record with highest likes is kept"""
    sortedbdf = dataframe.sort_values(by=['shortcode', 'likes'], ascending=False)
    dedup_df = sortedbdf.drop_duplicates(subset = ["shortcode"], keep = 'first')
    return dedup_df

def condition_labels(dataframe, column_name ='img_lables', exclude = False, exclude_tokens = EXCLUDE):
    """Removes words from img_labels column that matches exclude list"""
    dataframe[column_name] = dataframe[column_name].astype(str)
    dataframe[column_name] = dataframe[column_name].apply(lambda x: x.lower())
    if exclude:
        dataframe[column_name] = [' '.join([item for item in x.split() if item not in exclude_tokens]) for x in dataframe[column_name]]    
    else:
        pass
      
    return dataframe

"""
Depricated
def geo_coordinates(address):
    Utility to obtain geocoding for given address using Mapquest API services
    Parameters:
        Address - A string containing address to be geocoded
    Return:
        lattitude,longitude,geo_city,geo_state,geo_country
    import requests
    import json
    
    lattitude = ''
    longitude = ''
    geo_city = ''
    geo_state = ''
    geo_country = ''

    print('calling geo_coordinates for {address}')  
    parameters = {
        "key": MAPQKEY,
        "location": address
    }

    try:
        response = requests.get(MAPQURL, params=parameters)
    
        if response.status_code ==  200:                 
            jdata = json.loads(response.text)['results']              
            lattitude = jdata[0]['locations'][0]['latLng']['lat']
            longitude = jdata[0]['locations'][0]['latLng']['lng']
            geo_city = jdata[0]['locations'][0]['adminArea5']
            geo_state = jdata[0]['locations'][0]['adminArea3']
            geo_country = jdata[0]['locations'][0]['adminArea1'] 
    except:                
        str_address = address.split(',')
        if len(str_address)==1:
            geo_city = str_address[0]
        elif len(str_address)==2:
            geo_city = str_address[0]
            geo_country = str_address[1]
        elif len(str_address)==3:
            geo_city = str_address[0]
            geo_state = str_address[1]
            geo_country = str_address[2]
        else:   
            pass
    print('Success - calling geo_coordinates for {address}')  
    return lattitude,longitude,geo_city,geo_state,geo_country

"""


def geo_coordinates(datafile):
    """For one time use only to upgrade already downloaded data with geocodes"""
    
    import pandas as pd
    import requests
    import json
    
    data = pd.read_csv(datafile) 
    data.fillna('', inplace=True)
    
    for i, row in data.iterrows(): 
        address = ''    
        lattitude = ''
        longitude = ''
        geo_city = ''
        geo_state = ''
        geo_country = ''
        
        if row.city == '':
            if row.location == '':
                pass
            else:
                address = row.location
        else:
            address = row.city
            
        if address != '' and row.geo_country == '':          
            #print('calling mapquest API')
            parameters = {
                "key": MAPQKEY,
                "location": address
            }
        
            try:
                response = requests.get(MAPQURL, params=parameters)
            
                if response.status_code ==  200:                    
                    
                    jdata = json.loads(response.text)['results']    
                  
                    lattitude = jdata[0]['locations'][0]['latLng']['lat']
                    longitude = jdata[0]['locations'][0]['latLng']['lng']
                    geo_city = jdata[0]['locations'][0]['adminArea5']
                    geo_state = jdata[0]['locations'][0]['adminArea3']
                    geo_country = jdata[0]['locations'][0]['adminArea1']
            except Exception as e:
                print(e)
                str_address = address.split(',')
                if len(str_address)==1:
                    geo_city = str_address[0]
                elif len(str_address)==2:
                    geo_city = str_address[0]
                    geo_country = str_address[1]
                elif len(str_address)==3:
                    geo_city = str_address[0]
                    geo_state = str_address[1]
                    geo_country = str_address[2]
                else:   
                    pass
                    
            data.loc[i, 'lattitude'] = lattitude
            data.loc[i, 'longitude'] = longitude
            data.loc[i, 'geo_city'] = geo_city
            data.loc[i, 'geo_state'] = geo_state
            data.loc[i, 'geo_country'] = geo_country
            
            #print('Mapquest API Success')
    
    #Storing dataframe as csv files per hashtag
    data.to_csv(datafile, index = None)    
    
    print("Geocoded file path:",datafile)

def normalize_date(filepath):
    
    """to be used only for Bulk correction/normalozation of date format to normalize()
    function to normalize date column to post_date column
    Parameters:
        Data -> Dataframe
    Returns:
        Dataframe with normalized date
    """	
	
    import pandas as pd
    import os
    from pathlib import Path
	
    if os.path.isdir(filepath):
        print(f'Source Path exists: {filepath}')
        for file in Path(filepath).glob('*.csv'):    
            print('*******************************')
            print(f'Normalizing post_date for file {file} ........')
            data = pd.read_csv(file)
            data['post_date'] = pd.to_datetime(data['post_date']).dt.normalize()
            data.to_csv(file, index = None)
            print("Date Normalized for file at path:",file)
    else:
        print(f'Source Path does not exists: {filepath}')
            
def geo_coordinates_df(datafile):
    """For one time use only to upgrade already downloaded data with geocodes"""
    
    import pandas as pd
    import requests
    import json
    
    data = pd.read_csv(datafile)    
    data.fillna('', inplace=True)
    
    lattitude_list = []
    longitude_list = [] 
    geo_city_list = []
    geo_state_list = []
    geo_country_list = []
    
    for i, row in data.iterrows(): 
        address = ''    
        lattitude = ''
        longitude = ''
        geo_city = ''
        geo_state = ''
        geo_country = ''
        if row.city == '':
            if row.location == '':
                pass
            else:
                address = row.location
        else:
            address = row.city
        if address != '':  
            parameters = {
                "key": MAPQKEY,
                "location": address
            }
        
            try:
                response = requests.get(MAPQURL, params=parameters)
            
                if response.status_code ==  200:                    
                    
                    jdata = json.loads(response.text)['results']    
                  
                    lattitude = jdata[0]['locations'][0]['latLng']['lat']
                    longitude = jdata[0]['locations'][0]['latLng']['lng']
                    geo_city = jdata[0]['locations'][0]['adminArea5']
                    geo_state = jdata[0]['locations'][0]['adminArea3']
                    geo_country = jdata[0]['locations'][0]['adminArea1']
            except Exception as e:
                print(e)
                str_address = address.split(',')
                if len(str_address)==1:
                    geo_city = str_address[0]
                elif len(str_address)==2:
                    geo_city = str_address[0]
                    geo_country = str_address[1]
                elif len(str_address)==3:
                    geo_city = str_address[0]
                    geo_state = str_address[1]
                    geo_country = str_address[2]
                else:   
                    pass
                    
        lattitude_list.append(lattitude)
        longitude_list.append(longitude)
        geo_city_list.append(geo_city)
        geo_state_list.append(geo_state)
        geo_country_list.append(geo_country)                
       
    data.insert(loc = 10,column = 'lattitude', value = lattitude_list)
    data.insert(loc = 11,column = 'longitude', value = longitude_list)
    data.insert(loc = 12,column = 'geo_city', value = geo_city_list)
    data.insert(loc = 13,column = 'geo_state', value = geo_state_list)
    data.insert(loc = 14,column = 'geo_country', value = geo_country_list)
    
    #Storing dataframe as csv files per hashtag
    data.to_csv(datafile, index = None)    
    
    print("Geocoded file path:",datafile)
    
    return data