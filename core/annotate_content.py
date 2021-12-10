# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:08:18 2021

@author: nkushwaha
"""

import io
import os
import re
import argparse
import pandas as pd
from pathlib import Path
from config import GAPP_CRED, DATA_DIR, ANNOTATED_DIR, RAW_DATA_DIR

# Imports the Google Cloud client library
from google.cloud import vision



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GAPP_CRED

# Instantiates a client
client = vision.ImageAnnotatorClient()

def gca_annotate(file):   
    file_path, file_name= os.path.split(file)
    #print("file_path",file_path)
    print("file_name",file_name)  
    
    #print("GAPP_CRED",GAPP_CRED) 
    #print("DATA_DIR",DATA_DIR)  
    annotate_file_path = ANNOTATED_DIR   
    #print("annotate_file_path",annotate_file_path)  
    
    #assert os.path.isdir(annotate_file_path), f'path: {annotate_file_path} already exists'
    if os.path.isdir(annotate_file_path):
        pass
    else:
        os.mkdir(annotate_file_path)
        
    post_df = pd.read_csv(file)
    img_lable_list=[]
    img_lable_score=[]
    counter = 0
    for index, row in post_df.iterrows():
        try:
            # The name of the image file to annotate
            img_file = os.path.normpath(row['img_path'])
            
            # Loads the image into memory
            with io.open(img_file, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Performs label detection on the image file
            response = client.label_detection(image=image)
            labels = response.label_annotations                
            img_lable_list.append(' '.join([re.sub(' ', '-',label.description) for label in labels]))
            img_lable_score.append([[re.sub(' ', '-',label.description) ,label.score] for label in labels])
            counter += 1
        except Exception as e:
            print(e)
            img_lable_list.append('')
            img_lable_score.append('')
    post_df["img_lables"] = img_lable_list  
    post_df["img_lable_scores"] = img_lable_score 
    annotated_file = os.path.join(annotate_file_path,file_name)   
    post_df.to_csv(annotated_file, index = None)     
    print("Annotated file path:",annotated_file)  
    print('No of images succesfully annotated:',counter)  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data collection from Instagram Post data based on hashtag')
    parser.add_argument('-f', '--filepath', type=str, default=RAW_DATA_DIR, required=False,
                        help='File Path')
    
    args = parser.parse_args()
        
        
    #Checking existence of files for given hastag and calling collect_post funtion
    if os.path.isdir(args.filepath):
        print(f'Source Path exists: {args.filepath}')
        for file in Path(args.filepath).glob('*.csv'):    
            print('*******************************')
            print(f'Annotaing file {file} ........')
            gca_annotate(file)
    else:        
        print(f'Source Path does not exists: {args.filepath}')
    

