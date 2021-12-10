# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:52:40 2021

@author: nkushwaha
"""

import io
import os
import re
import argparse
import pandas as pd
from pathlib import Path
from config import GAPP_CRED, DATA_DIR, ANNOTATED_DIR, RAW_DATA_DIR

# for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.style.use('fivethirtyeight')

   
def word_graphics(dataframe):
    from wordcloud import WordCloud
    wordcloud = WordCloud(background_color = 'lightcyan',
                      width = 1200,
                      height = 500).generate(str(dataframe['img_lables']))

    plt.figure(figsize = (10, 10))
    plt.imshow(wordcloud)
    plt.title("WordCloud ", fontsize = 10)
    
def word_distribution(dataframe):   
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    import numpy as np

    cv = CountVectorizer(stop_words = 'english')
    words = cv.fit_transform(dataframe['img_lables'])
    sum_words = words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    
    color = plt.cm.twilight(np.linspace(0, 1, 20))
    frequency.head(50).plot(x='word', y='freq', kind='bar', figsize=(20, 7), color = color)
    plt.title("Most Frequently Occuring Words - Top 50")

    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data collection from Instagram Post data based on hashtag')
    parser.add_argument('-f', '--filepath', type=str, default=ANNOTATED_DIR, required=False,
                        help='File Path')
    
    args = parser.parse_args()
        
        
    #Checking existence of files for given hastag and calling collect_post funtion
    if os.path.isdir(args.filepath):
        print(f'Source Path exists: {args.filepath}')
        for file in Path(args.filepath).glob('*.csv'):    
            print('*******************************')
            print(f'Annotaing file {file} ........')
    else:        
        print(f'Source Path does not exists: {args.filepath}')    