# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:29:20 2021

@author: nkushwaha
"""
import argparse
import os 
import sys
from pathlib import Path
from config import PROJECT_ROOT_DIR, DATA_DIR,RAW_DATA_DIR,ANNOTATED_DIR, WEBDRIVER_LOC, BASE_URL, CSV_DIR

print(Path(os.getcwd()).resolve())
print(Path(__file__).resolve().parent.parent.parent)

#if Path(__file__).resolve().parent not in sys.path:
    #print('not present')

print(RAW_DATA_DIR)

p = Path(RAW_DATA_DIR)
print(Path(RAW_DATA_DIR).glob('*.csv'))
"""
files = list(p.glob('*.csv'))
for i in files:
    print(i)
"""    
for file in Path(RAW_DATA_DIR).glob('*.csv'):
    print(file)


"""
currentDirectory = Path('.')

for currentFile in currentDirectory.iterdir():
    print(currentFile)

    print(args.username)
    print(args.password)
    print(args.hashtag)
    print(args.basepath)
    print(Path(os.getcwd()).resolve().parent)    
    print('from config:', PROJECT_ROOT_DIR)      
    print('data dir from config:', DATA_DIR)
    """
    
 