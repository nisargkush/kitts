"""
Model Builder containing model and supporting utilities to build clustering and topic models
Classes:
    Tokenizer : Class containing methods to tolenize image lables
    SimpleVectorize : Class containing count vectorizer and TFIDF vectorizer to generate feature vectore 
                      and feature names
    Word2Vectorize :  Class containing word2vec vectorizer methods to train, call and use w2v model from 
                     image label tokens and generate feature vectore and feature names
    kittsCluster : Encapsulates Kmeans funtionality methods to train and visualize cluster models.
                   Provides utilities to train, store and visualize clustered features
    CorexModel : Corelation Explanation, topic model to train and visualization utility to create,
                 store and visualize clustered features using, semi-supervized anchoring techniques
Methods & Options:
    
"""

import os
#import argparse
import pandas as pd
from kitts.config import W2V_MODEL_FNAME, TOURISM_ANCHORS, CLUSTERED_DIR
#from sklearn import cluster

class Tokenizer():
    
    def __init__(self, dataframe, column_name = 'img_lables'):
        self.dataframe  =dataframe  
        if column_name == None:
            self.column_name = 'img_lables'
        else:
            self.column_name = column_name
    
    def tokenize(self):
        """To convert lables to tokens such that phares or composit words 
        with '-' remains as single word"""
        from nltk import word_tokenize
        
        tokenized_label = []
        for s in self.dataframe[self.column_name]:
            tokenized_label.append(word_tokenize(s.lower()))
        return tokenized_label

class SimpleVectorize():
    def __init__(self, dataframe, column_name = 'img_lables'):
        self.dataframe  =dataframe  
        if column_name == None:
            self.column_name = 'img_lables'
        else:
            self.column_name = column_name
        
        
    def count_vectorize(self, max_features=10000):
        from sklearn.feature_extraction.text import CountVectorizer
        
        Cvectorizer = CountVectorizer(stop_words='english', max_features= max_features, binary=True)
        feature_vectors = Cvectorizer.fit_transform(self.dataframe[self.column_name])
        
        return Cvectorizer , feature_vectors

    def tfidf_vectorize(self, max_features=10000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        Tvectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, binary=True)
        feature_vectors = Tvectorizer.fit_transform(self.dataframe[self.column_name])
        
        return Tvectorizer, feature_vectors

class Word2Vectorize():
    def __init__(self, dataframe, column_name = 'img_lables', w2vmodelf = W2V_MODEL_FNAME):
        if w2vmodelf == None:
            self.w2vmodelf = W2V_MODEL_FNAME
        else:
            self.w2vmodelf = w2vmodelf
        self.dataframe  =dataframe        
        if column_name == None:
            self.column_name = 'img_lables'
        else:
            self.column_name = column_name    
            
    def get_model(self):
        """funtion loads saved model based on constructor call"""
        import os
        from gensim.models import Word2Vec
 
        if os.path.isfile(self.w2vmodelf):
            smodel = Word2Vec.load(self.w2vmodelf)
        else:
            print(f'Model does not exist. Please create a new model and save at {self.w2vmodelf}')
        return smodel
    
    def train_model(self, tokenized_label, max_features):
        
        from gensim.models import Word2Vec
        smodel = self.get_model()
        model = Word2Vec(sentences=tokenized_label, vector_size= max_features, window=5, min_count=1, workers=4)
        model.build_vocab(tokenized_label, progress_per=1000)
        smodel.train(tokenized_label, total_examples=model.corpus_count, epochs=model.epochs)
        smodel.save(self.w2vmodelf)
        return smodel
    
    def get_similar_words(self, token):        
        """returns words similar to given word"""      
        smodel = self.get_model()
        return smodel.wv.most_similar(token)
    
    def vectorize(self,tokenized_label):
        """
        Generate vectors for list of documents using a trained w2v model
    
        Args:
            tokenized_labels: List of documents
            model: 
    
        Returns:
            List of token vectors
        """
        import numpy as np        
        model = self.get_model()
        feature_vectors = []
        feature_names = []
    
        for tokens in tokenized_label:
            zero_vector = np.zeros(model.vector_size)
            vectors = []
            for token in tokens:
                if token in model.wv:
                    try:
                        vectors.append(model.wv[token])
                    except KeyError:
                        continue
            if vectors:
                vectors = np.asarray(vectors)
                avg_vec = vectors.mean(axis=0)
                feature_vectors.append(avg_vec)
                feature_names.append(token)
            else:
                feature_vectors.append(zero_vector)                
                feature_names.append(token)
        return feature_names, feature_vectors   
        
    def display_wv_cluster_words(self, cluster, words_in_clusters):
        """Display top 'no_of_words' per cluster for a given 'cluster'"""
        
        smodel = self.get_model()
        print("Most representative terms per cluster (based on centroids):")
        cluster_words = []
        for i in range(len(cluster.cluster_centers_)):
            tokens_per_cluster = ""
            most_representative = smodel.wv.most_similar(positive=[cluster.cluster_centers_[i]], topn = words_in_clusters)
            cluster_words.append(most_representative)
            for t in most_representative:
                tokens_per_cluster += f"{t[0]} "
            print("***********************************")
            print(f"Cluster {i}: {tokens_per_cluster}")
        return cluster_words   

    
class kittsCluster():
    
        
    def __init__(self,feature_vectors, feature_names):
        self.feature_vectors = feature_vectors
        self.feature_names = feature_names
        
    def mbkmeans_clusters(self, n, batch_size, print_silhouette_val):
        """Generate clusters and print Silhouette metrics using MBKmeans    
        Args:
            n: Number of clusters.
            mb: Size of mini-batches.
            print_silhouette_values: Print silhouette values per cluster.    
        Returns:
            Trained clustering model and labels based on X.
        """
        
        from sklearn import cluster
        from sklearn.metrics import silhouette_samples, silhouette_score
        km = cluster.MiniBatchKMeans(n_clusters=n, batch_size=batch_size).fit(self.feature_vectors)
        print(f"For n_clusters = {n}")
        print(f"Silhouette coefficient: {silhouette_score(self.feature_vectors, km.labels_):0.2f}")
        print(f"Inertia:{km.inertia_}")
    
        if print_silhouette_val:
            sample_silhouette_values = silhouette_samples(self.feature_vectors, km.labels_)
            print("Silhouette values:")
            silhouette_values = []
            for i in range(n):
                cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
                silhouette_values.append(
                    (
                        i,
                        cluster_silhouette_values.shape[0],
                        cluster_silhouette_values.mean(),
                        cluster_silhouette_values.min(),
                        cluster_silhouette_values.max(),
                    )
                )
            silhouette_values = sorted(
                silhouette_values, key=lambda tup: tup[2], reverse=True
            )
            for s in silhouette_values:
                print(
                    f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
                )
        return km, km.labels_
    
    def display_cluster_words(self, cluster, words_in_clusters=25):
        """Display top 'no_of_words' per cluster for a given 'cluster'"""        
        
        print("Printing {words_in_clusters} per cluster (based on centroids):")
        order_centroids = cluster.cluster_centers_.argsort()[:, ::-1]
        terms = self.feature_names
        
        cluster_words = []
        for i in range(len(cluster.cluster_centers_)):
            print("Cluster %d:" % i)
            tokens_per_cluster = [terms[ind ]for ind in order_centroids[i, :words_in_clusters]]   
            cluster_words.append(tokens_per_cluster)
            print("***********************************")
            print(tokens_per_cluster)
        return cluster_words    
    
    def visualize_cluster_2D(self, cluster):
        """Display 2D representztion of clustered points using PCA"""        
        
        
        import pandas as pd
        from sklearn.decomposition import PCA #Principal Component Analysis
        import matplotlib.pyplot as plt
        
        #PCA with two principal components
        pca_2d = PCA(n_components=2)
        
        #This DataFrame contains the two principal components that will be used
        #for the 2-D visualization mentioned above
        PCs_2d = pd.DataFrame(pca_2d.fit_transform(self.feature_vectors.toarray()))
        
        PC2kmeans = cluster
        PC2kmeans.fit(PCs_2d)
        pred_cluster = PC2kmeans.predict(PCs_2d)
        centers = PC2kmeans.cluster_centers_
        #plotting points on 2D
        plt.scatter(PCs_2d[0], PCs_2d[1], c=pred_cluster, s=5, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1],c='black', s=50, alpha=0.6)
        plt.show()    
        
    def visualize_cluster_3D(self, cluster):
        """Display 3D representztion of clustered points using PCA"""        
        
        
        import pandas as pd
        from sklearn.decomposition import PCA #Principal Component Analysis
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        #PCA with two principal components
        pca_3d = PCA(n_components=3)
        
        #This DataFrame contains the two principal components that will be used
        #for the 2-D visualization mentioned above
        PCs_3d = pd.DataFrame(pca_3d.fit_transform(self.feature_vectors.toarray()))
        
        PC3kmeans = cluster
        PC3kmeans.fit(PCs_3d)
        pred_cluster = PC3kmeans.predict(PCs_3d)
        centers = PC3kmeans.cluster_centers_
        fig = plt.figure()
        ax = Axes3D(fig)
        #plotting points on 3D
        ax.scatter(PCs_3d[0], PCs_3d[1], PCs_3d[2], c=pred_cluster, s=5, cmap='viridis')
        ax.scatter3D(centers[:, 0], centers[:, 1],centers[:, 2],c='black', s=50, alpha=0.6)
        plt.show()     
    
    def visualize_wv_cluster_2D(self, cluster):
        """Display 2D representztion of clustered points using PCA"""        
        
        
        import pandas as pd
        from sklearn.decomposition import PCA #Principal Component Analysis
        import matplotlib.pyplot as plt
        from scipy.sparse import csr_matrix
        
        #PCA with two principal components
        pca_2d = PCA(n_components=2)
        
        #convering w2v vectors to csr matrix
        Wvectors = csr_matrix(self.feature_vectors)
        
        #This DataFrame contains the two principal components that will be used
        #for the 2-D visualization mentioned above
        PCs_2d = pd.DataFrame(pca_2d.fit_transform(Wvectors.toarray()))
        
        PC2kmeans = cluster
        PC2kmeans.fit(PCs_2d)
        pred_cluster = PC2kmeans.predict(PCs_2d)
        centers = PC2kmeans.cluster_centers_
        #plotting points on 2D
        plt.scatter(PCs_2d[0], PCs_2d[1], c=pred_cluster, s=5, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1],c='black', s=50, alpha=0.6)
        plt.show()    
        
    def visualize_wv_cluster_3D(self, cluster):
        """Display 3D representztion of clustered points using PCA"""        
        
        
        import pandas as pd
        from sklearn.decomposition import PCA #Principal Component Analysis
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.sparse import csr_matrix
        
        #PCA with two principal components
        pca_3d = PCA(n_components=3)
        
        #convering w2v vectors to csr matrix
        Wvectors = csr_matrix(self.feature_vectors)
        
        #This DataFrame contains the two principal components that will be used
        #for the 2-D visualization mentioned above
        PCs_3d = pd.DataFrame(pca_3d.fit_transform(Wvectors.toarray()))
        
        PC3kmeans = cluster
        PC3kmeans.fit(PCs_3d)
        pred_cluster = PC3kmeans.predict(PCs_3d)
        centers = PC3kmeans.cluster_centers_
        fig = plt.figure()
        ax = Axes3D(fig)
        #plotting points on 3D
        ax.scatter(PCs_3d[0], PCs_3d[1], PCs_3d[2], c=pred_cluster, s=5, cmap='viridis')
        ax.scatter3D(centers[:, 0], centers[:, 1],centers[:, 2],c='black', s=50, alpha=0.6)
        plt.show()
        
    def clustered_data(self, cluster, data, fpath = CLUSTERED_DIR):
        """
        funtion will create a dataframe such that each data point 
        is annotated with one cluster number and store in any given path
        Parameters
        cluster : kmeans cluster model
        fpath : path to store clustered csv, defaulted to CLUSTERED_DIR
        """
        #converting date to date only(without timestamp) format
        data['post_date'] = pd.to_datetime(data['post_date']).dt.normalize()
        
        #Appending cluster to data(frame) and writing it to .csv file
        data['clusters'] = cluster.labels_
        file_path = os.path.join(fpath,'kmeans_Cluster_Data.csv')
        data.to_csv(file_path,index=False)
        print(f'file created and saved at {file_path}')
            
class CorexModel():
    
    
    def __init__(self,feature_vectors, feature_names, no_of_topics, anchors = TOURISM_ANCHORS, anchor_strength = 3):
        self.feature_vectors = feature_vectors
        self.feature_names = feature_names 
        self.no_of_topics = no_of_topics
        if anchors is None:
            self.anchors = []
        else:
            self.anchors = TOURISM_ANCHORS
        if anchor_strength == '':
            self.anchor_strength = 3
        
        
    def corex(self, ):
        """Created CoRex topic model based on given data and parameetrs
        uses anchor words to incline topic words to suggested topics
        Parameters:
        n_hidden: number of topics ("hidden" as in "hidden latent topics")
        words: words that label the columns of the doc-word matrix (optional)
        docs: document labels that label the rows of the doc-word matrix (optional)
        max_iter: number of iterations to run through the update equations (optional, defaults to 200)
        verbose: if verbose=1, then CorEx will print the topic TCs with each iteration
        seed: random number seed to use for model initialization (optional)    
        """
        import corextopic.corextopic as ct
        
        words = [word for ind,word in enumerate(self.feature_names)]
        topic_model = ct.Corex(n_hidden=self.no_of_topics, words=words, max_iter=200, verbose=False, seed=1)
        topic_model.fit(self.feature_vectors, words=words, anchors=self.anchors, anchor_strength=5)
        
        return topic_model, words
        
    def corex_topics(self, topic_model, no_of_words):
         
        """
        Print all topics from the CorEx topic model
        Also returs a list of 3 tuple list per topic
        """
        #topic_model.get_topics(topic=5, n_words=10, print_words=True)
        topics = topic_model.get_topics(n_words=no_of_words)
        for n,topic in enumerate(topics):
            try:
                topic_words,_,_ = zip(*topic)
                #print(topic)
                print('{}: '.format(n) + ', '.join(topic_words))
            except:
                print(n,':',topic)
                
        return topics
    
    def plot_corex(self, topic_model):
        """Plots total correlation v/s topics"""
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
        plt.xlabel('Topic', fontsize=16)
        plt.ylabel('Total Correlation (nats)', fontsize=16)
        plt.show()
        
    def corex_report(self, topic_model, words):
                           
        """Output repository created at working directory"""
        
        import corextopic.vis_topic as vt
        vt.vis_rep(corex = topic_model, column_label=words, prefix = 'corex_visuals' )
        
    def convert_lable(self, value, column):
        "to be used as lambda function to covert True/False values in corex model lables to column name"
        if value:
            return column
        else:
            return ''
        
    def clustered_data(self, topic_model, data, cluster_columns = TOURISM_ANCHORS, fpath = CLUSTERED_DIR):
        """
        funtion will create a dataframe such that each data point 
        is annotated with one or more toipics based on model
        and store in any given path
        Parameters
        topic_model : corex model
        cluster_columns : Provide list of topics, defaulted to TOURISM_ANCHORS
        fpath : path to store clustered csv, defaulted to CLUSTERED_DIR
        """
        #generating topic columns to be added to dataframe
        if len(cluster_columns) < len(topic_model.labels[0]):
            for i in range (len(cluster_columns), len(topic_model.labels[0])):
                cluster_columns.append(f'extra{i+1}')   
        
        #Dropping irrelavant columns
        data = data.drop(['post_url','like_and_view_counts_disabled','is_paid_partnership',
                          'exact_city_match','is_video','is_verified','is_private','img_lable_scores'],
                         axis =1)
                        
        #converting date to format 'YYYY-MM-DD'
        data['post_date'] = pd.to_datetime(data['post_date']).dt.normalize()
        
        #Appending cluster to data(frame) with Topic keywords
        data[cluster_columns] = topic_model.labels
        for column in cluster_columns:
            data[column] = data[column].apply(lambda val: self.convert_lable(val, column)) 
            
        #creating concatenation of labels as column 'combined_labels'
        for index, rows in data.iterrows(): 
            s = ''
            for column in cluster_columns:
                if data.at[index,column] == '':
                    pass
                else:
                    s = s + ' ' + data.at[index,column]
            data.at[index,'combined_labels'] = s.strip()
        
        file_path = os.path.join(fpath,'Corex_Cluster_Data.csv')
        data.to_csv(file_path,index=False)
        print(f'file created and saved at {file_path}')

class ClusterMap():       
    
    def __init__(self,features):
        self.features = features
        
    def elbow_plot(self, max_cluster):
        """
        Plots elbow graph to visualize ideal number of cluster based on inertia
        """
        import sys
        from sklearn.cluster import KMeans
               
        wcss = []
        for i in range(1, max_cluster, 5):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(self.features)
            wcss.append(kmeans.inertia_)
            print('\rProgress: %d' % i, end='')
            sys.stdout.flush()
        
        import matplotlib.pyplot as plt
        plt.plot(range(1, max_cluster, 5), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        
        
    def silhouette_plot(self, clusters_i):
        """
        Displays Silhouette Visulaizer for K means clutering algorithm
        for a range of clusters plotted in 2D plot
        Parameter - 
        clusters_i = list of number of clusters to compare.
                    Accepts even number of clusters to compare.
                    Else pads with a random number between 1-20
        """
        
        from yellowbrick.cluster import SilhouetteVisualizer
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        import numpy as np
        import random 
        
               
        try:
            npcluster = np.array(clusters_i).reshape(-1,2)
        except ValueError:
            clusters_i.append(random.randint(1,20))
            npcluster = np.array(clusters_i).reshape(-1,2)
            
        r, c = np.array(npcluster).reshape(-1,2).shape
        
        fig, ax = plt.subplots(r, c, figsize=(15,8))
        for i in clusters_i:
            '''
            Create KMeans instance for different number of clusters
            '''
            km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
            
            #getting positions to plot silhouette from 2Dnumpy array
            row, column = np.where(npcluster==i)
            plotr = row[0]
            plotc = column[0]
            
            #q, modc = divmod(i, c)
            #p, modr = divmod(q, r)
            '''
            Create SilhouetteVisualizer instance with KMeans instance
            Fit the visualizer
            '''
            if r==1:#to handle 1D plot figure array, if only 2 values are provided in param
                ax[plotc].set_title(f'n = {i} Clusters')                
                visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[plotc])
            else:
                ax[plotr, plotc].set_title(f'n = {i} Clusters')
                visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[plotr][plotc])
            visualizer.fit(self.features)
        
    def findOptimalEps(self, n_neighbors):
        '''
        function to find optimal eps distance when using DBSCAN        
        '''
        from sklearn.neighbors import NearestNeighbors # for selecting the optimal eps value when using DBSCAN
        import numpy as np
        import matplotlib.pyplot as plt
        
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(self.features)
        distances, indices = nbrs.kneighbors(self.features)
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