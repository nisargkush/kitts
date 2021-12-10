# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:46:23 2021

@author: nkushwaha
"""

import io
import os
import re
import argparse
import pandas as pd
from pathlib import Path
from config import GAPP_CRED, DATA_DIR, ANNOTATED_DIR, RAW_DATA_DIR


def accumulate_dataframe(filepath):
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
            return Bigframe
        except Exception as e:
            print(e)
    else:        
        print(f'Source Path does not exists: {filepath}')
        
def dedup_dataframe(dataframe):
    sortedbdf = dataframe.sort_values(by=['shortcode', 'likes'], ascending=False)
    dedup_df = sortedbdf.drop_duplicates(subset = ["shortcode"], keep = 'first')
    return dedup_df
        
def cleanse_lable(dataframe, column, lower=True, ascii_chars=True, no_numbers=True, no_punctuation=True, remove_stopwords=True, lemmatize=True, custom_blank_text='non ascii symbols punctuations numbers'):
    
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk    
    import string
    
    #Lower case
    if lower == True:
        dataframe['Query_Modified'] = dataframe[column].str.lower()
    
    #Remove non-ascii characters
    if ascii_chars == True:                            
        dataframe["Clean_Lables"] = dataframe["Clean_Lables"].apply(lambda x: ''.join([" " if i not in string.printable else i for i in x]))
    
    #Remove numbers
    if no_numbers == True:
        dataframe['Clean_Lables'] = dataframe['Clean_Lables'].str.replace(r'\d', '')
    
    #Punctuation
    if no_punctuation == True:
        dataframe['Clean_Lables'] = dataframe['Clean_Lables'].str.replace(r'[^\w\s]+', ' ')
    
    #Remove stopwords
    if remove_stopwords == True:
        stop = stopwords.words('english')
        dataframe['Clean_Lables'] = dataframe['Clean_Lables'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    #Lemmatize words
    if lemmatize == True:
        wnl = WordNetLemmatizer()
        def lemmatize_all(sentence):
            text = list()
            for word, tag in nltk.pos_tag(str.split(sentence)):
                if tag.startswith("NN"):
                    text.append( wnl.lemmatize(word, pos='n'))
                elif tag.startswith('VB'):
                    text.append( wnl.lemmatize(word, pos='v'))
                elif tag.startswith('JJ'):
                    text.append( wnl.lemmatize(word, pos='a'))
                else:
                    text.append( word)
            return ' '.join(text)            

        dataframe['Clean_Lables'] = dataframe['Clean_Lables'].apply(lambda sentence: ' '.join([lemmatize_all(sentence)]))
    
    #Replacing blanks from ascii characters, punctuations and numbers with custom text
    dataframe['Clean_Lables'].replace(r'^\s*$', custom_blank_text, regex=True, inplace = True)
    
    #Extra Spaces
    dataframe['Clean_Lables'] = dataframe['Clean_Lables'].apply(lambda x: re.sub("\s\s+", " ", str(x.strip())))
    
    print('Done')
    
    return dataframe      