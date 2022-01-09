"""
Generic data visulaization methods
Usage:
    Used inside other project methods. Can also be directly called from notebook    
Methods:
    word_graphics : Displays image labels in word cloud format
    word_distribution : Displays top 50 image labes and it's frequency
    showimlable : Shows a random image from storage along with it's labels
"""

import os
import pandas as pd
#import argparse

# for visualization
#%pylab inline
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.style.use('fivethirtyeight')

   
def word_graphics(dataframe):
    """
    Displays image labels in word cloud format
    Parameters:
        dataframe: dataframe containing 'img_lables' column to visualize
    Returns:
        None
    """
    from wordcloud import WordCloud
    wordcloud = WordCloud(background_color = 'lightcyan',
                      width = 1200,
                      height = 500).generate(str(dataframe['img_lables']))

    plt.figure(figsize = (10, 10))
    plt.imshow(wordcloud)
    plt.title("WordCloud ", fontsize = 10)
    
def word_distribution(dataframe):  
    """
    Displays top 50 image labes and it's frequency
    Parameters:
        dataframe: dataframe containing 'img_lables' column to visualize
    Returns:
        None
    """
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
    
def showimlable(dataframe, n):
    """
    Shows a random image from storage along with it's labels
    Parameters:
        dataframe: dataframe containing 'img_lables' column to visualize
        n: Upper limit of random number of images in store
    Returns:
        None
    """
    i = random.randint(0,n)
    data = dataframe.loc[i,['post_date','city','img_lables','img_path']]
    plt.figure(figsize=(4,4))
    plt.grid(False)
    img = mpimg.imread(os.path.normpath(data['img_path']))
    plt.imshow(img)
    plt.show()
    print ('Post Date: ',data['post_date'])
    print ('City: ',data['city'])
    print ('Image Labels: ',data['img_lables'])
    