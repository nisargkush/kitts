# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:31:31 2021

@author: nkushwaha
"""

import os
import wget

#def write_csv()
base_path = os.getcwd()
print(base_path)
pathd = os.path.join(base_path,"italytravel",)
#os.mkdir(pathd)
file_name = os.path.join(pathd,"italytravel") + ".csv"
print(file_name)
bpath, hashtag= os.path.split(file_name)
print(bpath)
print(hashtag)
