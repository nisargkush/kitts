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
from config import GAPP_CRED, DATA_DIR, ANNOTATED_DIR, RAW_DATA_DIR, GlOVE_FILE_PATH, GLOVE_FILE, CLUSTERED_DIR


def count_vectorize(dataframe, column_name, max_features):
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features, binary=True)
    doc_word = vectorizer.fit_transform(dataframe[column_name])
    
    return doc_word

def tfidf_vectorize(dataframe, column_name, max_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, binary=True)
    doc_word = vectorizer.fit_transform(dataframe[column_name])
    
    return doc_word
    
def glove_vectorize(dataframe, column, method, n=3):
    import numpy as np
    import sys
    if method=="first_n_words":
        #Approach: First n words
        #Add 'blank' to words less than n
        dataframe['Length_Glove_Words'] = dataframe[column].str.split().str.len()
        def blank_words (row, n):
           for i in range(1,n+1) :
               if row['Length_Glove_Words'] == i :
                   return ' blank' * (n-i)
        
        dataframe['Words'] = dataframe.apply(lambda row: blank_words(row,n),axis=1)
        dataframe['Top_Words'] = dataframe[column].fillna('') + dataframe['Words'].fillna('')
        dataframe.drop(['Length_Glove_Words','Words'], axis=1, inplace=True)
        
        #Select First n Words
        dataframe['Top_Words'] = dataframe['Top_Words'].str.split().str[0:n].str.join(' ')
        
        #Add Glove embeddings
        #GLOVE_FILE = "glove.6B.100d.txt"
        Glovewords = pd.read_table(GlOVE_FILE_PATH+GLOVE_FILE, sep=" ", index_col=0, header=None, quoting=3)
        
        # Unique words
        unique = list(dataframe['Top_Words'].str.split(' ', expand=True).stack().unique())
        unique_word_vec=Glovewords.loc[unique].T.to_dict('list')
        del Glovewords,unique
        
        #Glove vectors for top 3 words        
        j=0
        length = len(dataframe)-1
        stack = list()
        for index, row in dataframe.iterrows():
            df = []
            for i in range(0,n):
                df = np.append(df,unique_word_vec[row.Top_Words.split(' ')[i]])
            stack.extend(np.vstack(df).T)
            if j==length:
               print('\rProgress:  100%', end='')
               sys.stdout.flush()
            elif j%100==0:
               print('\rProgress: %d' % j, end='')
               sys.stdout.flush()
            j+=1                        
        
        del unique_word_vec
        
        stack=pd.DataFrame(stack)
        
        cluster_dataset = dataframe[["Top_Words"]]
        cluster_dataset = pd.concat([cluster_dataset.reset_index(drop=True), stack], axis=1)
        del stack
        return cluster_dataset
    
    elif method == "sum_word_vectors":        
        #Approach: Sum of d word vectors for n words
        #Add Glove embeddings
        #GLOVE_FILE = "glove.6B.100d.txt"
        Glovewords = pd.read_table(GlOVE_FILE_PATH+GLOVE_FILE, sep=" ", index_col=0, header=None, quoting=3)
        
        # Unique words
        unique = list(dataframe[column].str.split(' ', expand=True).stack().unique())
        unique_word_vec=Glovewords.loc[unique].T.to_dict('list')
        del Glovewords,unique
        
        #Sum of Glove vectors for n words
        from operator import add
        j=0
        length = len(dataframe)-1
        stack = list()
        for index, row in dataframe.iterrows():
            sum_word_vec = [0]*100
            for word in row[column].split(' '):
                word_vec = unique_word_vec[word]
                sum_word_vec = list(map(add, sum_word_vec, word_vec))
            stack.extend([sum_word_vec])
            if j==length:
               print('\rProgress:  100%', end='')
               sys.stdout.flush()
            elif j%100==0:
               print('\rProgress: %d' % j, end='')
               sys.stdout.flush()
            j+=1
        
        stack=pd.DataFrame(stack)
        
        cluster_dataset = dataframe[[column]]
        cluster_dataset = pd.concat([cluster_dataset.reset_index(drop=True), stack], axis=1)
        del stack
        return cluster_dataset

def get_non_glove_words(dataframe, column, model):

    # Unique Words
    counts = dataframe[column].str.split(expand=True).stack().value_counts(dropna=False).rename_axis('unique_words').reset_index(name='counts')
    
    # Extracting Glove Words and Non Glove Words
    non_glove_words = list()
    glove_words = list()
    for i in counts['unique_words']:
        try:
            model.get_vector(i)
        except KeyError:
            non_glove_words.append(i)
        else:
            glove_words.append(i)
    
    #Non-Glove words
    non_glove_words_df = pd.DataFrame({'unique_non_glove_words':non_glove_words})
    non_glove_words_df = pd.merge(non_glove_words_df,counts,how='left',left_on=['unique_non_glove_words'],right_on=['unique_words']).iloc[:,[0,2]]
    non_glove_words_df['cum_perc'] = round(100*non_glove_words_df["counts"].cumsum()/non_glove_words_df["counts"].sum(),2)
    
    print('Done')
    
    return(non_glove_words_df)    

def replace_non_glove_words(data, non_glove_words_df, column):
    #Replacing Non Glove Words with Blanks    
    import sys
    j=0
    length = len(non_glove_words_df['unique_non_glove_words'])-1
    for i in non_glove_words_df['unique_non_glove_words']:
        data[column].replace(r'(\b)+%s+(\b)'%i, ' ', regex=True, inplace=True)
        if j==length:
            print('\rProgress:  100%', end='')
            sys.stdout.flush()   
        elif j%10==0:
            print('\rProgress: %d' % j, end='')
            sys.stdout.flush()
        j+=1
        
    #Extra Spaces
    data[column] = data[column].apply(lambda x: re.sub("\s\s+", " ", str(x.strip()))) 
    
    return data


    
def plot_elbow(dataframe,column_name, max_cluster):
    import sys
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(dataframe[column_name])
    #del cluster_dataset
    
    wcss = []
    for i in range(1, max_cluster, 5):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_)
        print('\rProgress: %d' % i, end='')
        sys.stdout.flush()
    
    import matplotlib.pyplot as plt
    plt.plot(range(1, max_cluster, 5), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
def silhouettePlot(range_, data):
    '''
    we will use this function to plot a silhouette plot that helps us to evaluate the cohesion in clusters (k-means only)
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from yellowbrick.cluster import SilhouetteVisualizer    

    from sklearn.cluster import KMeans
    half_length = int(len(range_)/2)
    range_list = list(range_)
    fig, ax = plt.subplots(half_length, 2, figsize=(15,8))
    for _ in range_:
        kmeans = KMeans(n_clusters=_, random_state=42)
        q, mod = divmod(_ - range_list[0], 2)
        sv = SilhouetteVisualizer(kmeans, colors="yellowbrick", ax=ax[q][mod])
        ax[q][mod].set_title("Silhouette Plot with n={} Cluster".format(_))
        sv.fit(data)
    fig.tight_layout()
    fig.show()
    fig.savefig("silhouette_plot.png")
    
def findOptimalEps(n_neighbors, data):
    '''
    function to find optimal eps distance when using DBSCAN; based on this article: https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
    '''
    from sklearn.neighbors import NearestNeighbors # for selecting the optimal eps value when using DBSCAN
    import numpy as np
    import matplotlib.pyplot as plt
    
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)  
    
#def dbscan()    
    
def progressiveFeatureSelection(df, n_clusters=3, max_features=4,):
    '''
    very basic implementation of an algorithm for feature selection (unsupervised clustering); inspired by this post: https://datascience.stackexchange.com/questions/67040/how-to-do-feature-selection-for-clustering-and-implement-it-in-python
    '''
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    feature_list = list(df.columns)
    selected_features = list()
    # select starting feature
    initial_feature = ""
    high_score = 0
    for feature in feature_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data_ = df[feature]
        labels = kmeans.fit_predict(data_.to_frame())
        score_ = silhouette_score(data_.to_frame(), labels)
        print("Proposed new feature {} with score {}". format(feature, score_))
        if score_ >= high_score:
            initial_feature = feature
            high_score = score_
    print("The initial feature is {} with a silhouette score of {}.".format(initial_feature, high_score))
    feature_list.remove(initial_feature)
    selected_features.append(initial_feature)
    for _ in range(max_features-1):
        high_score = 0
        selected_feature = ""
        print("Starting selection {}...".format(_))
        for feature in feature_list:
            selection_ = selected_features.copy()
            selection_.append(feature)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data_ = df[selection_]
            labels = kmeans.fit_predict(data_)
            score_ = silhouette_score(data_, labels)
            print("Proposed new feature {} with score {}". format(feature, score_))
            if score_ > high_score:
                selected_feature = feature
                high_score = score_
        selected_features.append(selected_feature)
        feature_list.remove(selected_feature)
        print("Selected new feature {} with score {}". format(selected_feature, high_score))
    return selected_features    
    
#Summarize results
def summary(dataframe, cluster_column, aggr_column, original_column, modified_column, top_n, show_original=False):
    
    if cluster_column == 'edge_media_preview_like':
        df = dataframe.groupby([cluster_column])[[aggr_column]].sum()
    elif cluster_column in ('post_date'):
        df = dataframe.groupby([cluster_column])[[pd.to_datetime(dataframe[aggr_column]).dt.date]].count()
    elif cluster_column in ('location'):
        df = dataframe.groupby([cluster_column])[[aggr_column]].count()
  
    
    if show_original==True:
        original_keywords = list()
        for i, row in df.iterrows():
            kws = ",".join(pd.Series(dataframe.loc[dataframe[cluster_column] == i,[original_column]].values.flatten()).str.split(expand=True).stack().value_counts(dropna=False).rename_axis('unique_words').reset_index(name='counts').loc[0:top_n-1,'unique_words'].tolist())
            original_keywords.extend([kws])        
        original_keywords = pd.DataFrame(original_keywords, columns=['Top Original Keywords'])
        df = pd.concat([df, original_keywords.reset_index(drop=True)], axis=1)
    
    modified_keywords = list()
    for i, row in df.iterrows():
        kws = ",".join(pd.Series(dataframe.loc[dataframe[cluster_column] == i,[modified_column]].values.flatten()).str.split(expand=True).stack().value_counts(dropna=False).rename_axis('unique_words').reset_index(name='counts').loc[0:top_n-1,'unique_words'].tolist())
        modified_keywords.extend([kws])        
    modified_keywords = pd.DataFrame(modified_keywords, columns=['Top Modified Keywords'])
    df = pd.concat([df, modified_keywords.reset_index(drop=True)], axis=1)
    df = df.round(4)
    
    return df  

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
            pass
    else:        
        print(f'Source Path does not exists: {args.filepath}')    