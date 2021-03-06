"""
Instagram data collection utillities Annotater
Usage:
    Used inside other project methods. Can also be directly called from notebook    
Methods:
    accumulate_dataframe : Accumulates CSV file in mentoined path to a composit Dataframe
    dedup_dataframe : Removes duplicate data based on shortcode such that record with highest likes is kept
    condition_labels : Removes words from img_labels column that matches exclude list
    azure_upload : Utility to upload Scrapped Images on Azure Storage container and as Blob with anonymous read access
    geo_coordinates : Utliliyy to enrich collected post data with geocodes - Lattitude & Longitude and normalize city and country names
"""

import os
#import argparse
import pandas as pd
from pathlib import Path
from kitts.config import DATA_FILES_DIR, ANNOTATED_DIR, EXCLUDE, MAPQKEY, MAPQURL, AZ_CONN_STRING, AZ_CONTAINER


def accumulate_dataframe(filepath = ANNOTATED_DIR):
    """Accumulates CSV file in mentoined path to a composit Dataframe
    Parameters:
        filepath : Path of files to be collected and combined as a big dataframe
    Returns:
        Combined dataframe to be used by ML model
    """    
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
    """Removes duplicate data based on shortcode such that record with highest likes is kept
    Parameters:
        dataframe : dataframe , that may contain duplicate posts
    Returns:
        dataframe : deduped dataframe based on logic mentioned above
    """
    sortedbdf = dataframe.sort_values(by=['shortcode', 'likes'], ascending=False)
    dedup_df = sortedbdf.drop_duplicates(subset = ["shortcode"], keep = 'first')
    return dedup_df

def condition_labels(dataframe, column_name ='img_lables', exclude = False, exclude_tokens = EXCLUDE):
    """Removes words from img_labels column that matches exclude list, so that do not cloud Model features with ubiquitous words
    Parameters:
        dataframe : dataframe, that may contain upper/camel format words or words to be excludes
    Returns:
        dataframe : dataframe, with words removed based on exclude list and lower formated
    """
    dataframe[column_name] = dataframe[column_name].astype(str)
    dataframe[column_name] = dataframe[column_name].apply(lambda x: x.lower())
    if exclude:
        dataframe[column_name] = [' '.join([item for item in x.split() if item not in exclude_tokens]) for x in dataframe[column_name]]    
    else:
        pass
      
    return dataframe

def azure_upload(file = '', bulk_upload =  False, fpath = DATA_FILES_DIR ):       
    """Utility to upload Scrapped Images on Azure Storage container and as Blob with anonymous read access
    Parameters:
        file : If populated will upload given file to AZ_CONTAINER
		bulk_upload : Boolean, to check single file upload or bulk file upload
		fpath : Path containg blob file to be uploaded in bulk
    Returns:
        None
    """

    from azure.storage.blob import BlobServiceClient
    import os
    
    
    connection_string = AZ_CONN_STRING
    container_name = AZ_CONTAINER
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    if not bulk_upload:  
        try:
                   
            file_path, file_name= os.path.split(file)               
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
            with open(file, "rb") as data:
                blob_client.upload_blob(data)
                print(f'uploaded file {file_name}')
        except Exception as e:
            print(f'Exception while uploading single file {e}')
    else:        
        if os.path.isdir(fpath):
            print(f'Source Path exists: {fpath}')
            counter = 0
            try:            
                for ifile in Path(fpath).glob('*.jpg'):                                
                    file_path, file_name= os.path.split(ifile)
                    try:                        
                        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
                        with open(ifile, "rb") as data:
                            blob_client.upload_blob(data)
                        counter += 1
                    except:
                        pass
            except Exception as e:                
                print(e)
            print(f'Tota Number of files uploaded {counter}')            
        else:        
            print(f'Source Path does not exists: {fpath}')


def geo_coordinates(datafile):
    """Utlility to enrich collected post data with geocodes - Lattitude & Longitude and normalize city and country names
    Parameters:
        datafile : Path of file to be gecoded
    Returns:
        datafile : Geocoded data file path
    """
    
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
    """Normalizes date format to date only format and saves csv file back to same path
    function to normalize date column to post_date column
    Parameters:
        filepath : Path of files to be date-normalized
    Returns:
        None
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
    """For one time use only to upgrade already downloaded csv file that do not contain geocoding columns with geocodes
    Parameters:
        datafile : Path of file to be gecoded
    Returns:
        data : Geocoded dataframe to display
    """
    
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