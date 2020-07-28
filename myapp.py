import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Button
from bokeh.plotting import figure
import pandas as pd, matplotlib.pyplot as plt
from gensim import corpora, models
from nltk.stem import SnowballStemmer
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import row, column
from bokeh.transform import factor_cmap
from bokeh.palettes import *
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
#from pprint import pprint
#import random as rd
import umap
from collections import Counter
from sklearn import metrics
import io
import requests
    
lang  = 'german'
stopw = stopwords.words(lang)
add_stopw = ['zb','evtl','ggf','vs', 'ja','bspw','bzw','bzgl','ggfs','sollte','sollten','es','Es']
add_buzzwords = ['runde','eigenes','thema','erforderlich','notwendig']
stopw = stopw + add_stopw + add_buzzwords
vect = CountVectorizer(stop_words = stopw)
wpt = nltk.WordPunctTokenizer()

ppath= 'C:/Users/chris/OneDrive/02_consulting projekte/Bridge/01_200720 ML Text Clustering/5_heroku_server'
  
##########################
''' Functions '''
##########################
def stop_word_removal(x):
    token = x.split()
    return ' '.join([w for w in token if not w in stopw])

def clean_text(doc):
    punctuation_signs = list('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~€0123456789„') #remove special characters, punctation, quotes, 
    for punct_sign in punctuation_signs:
        doc = doc.str.replace(punct_sign, '')
        doc = doc.str.lower() #all lower case
        doc = doc.str.strip() # remove empty space
        # doc = doc.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) # remove short words
        doc = doc.apply(stop_word_removal) # remove stop words
        return doc

def stem_snow(x):
    s_text = []
    stemmer = SnowballStemmer(language=lang)  
    for word in x:
        s_text.append(stemmer.stem(word))
    s_text = ''.join(s_text)
    return s_text

def stem_cistem(x):
    from nltk.stem.cistem import Cistem
    stemmer = Cistem() 
    s_text = []
    for word in x:
        s_text.append(stemmer.stem(word))
    s_text = ''.join(s_text)
    return s_text

def pos_clean(x):
    from pattern.de import parse 
    s = parse('Hello Tschuss', chunks=True,tagset="STTS", relations=True, lemmata=True).split()[0]
    sen=[]
    for i in s:
        if i[1] == 'NN' or i[1] == 'ADJA' or i[1] == 'FM' or i[1] == 'ADJD' or i[1] == 'APPRART':
            sen.append(i[0])
    return ' '.join(sen)

def lda_ts(x, num_topics, pas, alp, et):
    dictionary_LDA = corpora.Dictionary(x.str.split())
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in x.str.split()]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary_LDA, passes=pas, alpha=[alp]*num_topics, eta=[et]*len(dictionary_LDA.keys()),)
    return lda_model
    
def prep_tfidf(x):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(x)
    tv_matrix = tv_matrix.toarray()
    vocab = tv.get_feature_names()
    df2 = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
    return df2

## top 5 words per topic

def top5():
    topics = list(set(df.topic_umap_km)); top_text = []
    for i in range(len(topics)): top_text.append(' '.join(df.loc[df.topic_umap_km == topics[i], 'Idea']))
    punctuation_signs = list('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~€0123456789„') 
    '''
    tt = []
    token = 
    for j in punctuation_signs:
        [w for w in token if not w in j]'''
    top_text_p = []
    for i in range(len(top_text)):
        token = top_text[i].split() 
        top_text_p.append(' '.join([w for w in token if not w in stopw]))
    top_text = top_text_p
    for i in punctuation_signs: top_text = [w.replace(i,'') for w in top_text]
    add_stopw = ['zb','evtl','ggf','vs', 'ja','JA','Ja','bspw','bzw','bzgl','ggfs','sollte','sollten','es','Es','Soll',"Ein","zu","für","den","Den","Ist","Wie","wie",'zB','Themen']
    add_buzzwords = ['runde','eigenes','thema','erforderlich','notwendig']
    stopw2 = stopw + add_stopw + add_buzzwords
    top_text_c=[]
    for i in range(len(top_text)):
        token = top_text[i].split() 
        top_text_c.append(' '.join([w for w in token if not w in stopw2]))
    top_text = top_text_c
    tsplit=[]
    for i in top_text: tsplit.append(i.split())
    cwords=[]
    for i in range(len(tsplit)): cwords.append(pd.DataFrame(dict(Counter(tsplit[i])).items(), columns=['word','freq']).sort_values(by=['freq'], inplace = False, ascending = False))
    twords=[]
    for i in cwords: twords.append(list(i.word.head(5)))
    topic_dict = dict(zip(topics,twords))
    df.loc[:,'topic_top5w'] = ''
    df.loc[:,'topic_top5w'] = df['topic_top5w'].astype(object)
    for i in range(len(df)): df.at[i,'topic_top5w'] = topic_dict[df.topic_umap_km[i]]

###############################
'''Load data, some cleaning'''
###############################
df = pd.read_csv('https://raw.githubusercontent.com/Cand1d/umap-vis-ideas/master/0_Data/bridge/de/bertrandt.csv',sep='|')
df.loc['Date'] = df.Date.apply(pd.to_datetime, errors='coerce')
df = df.dropna() 
df.index = range(len(df))

############################
''' Prepare data '''
############################
df['c_text']  = clean_text(df.Idea)
df['c_text_snow'] = df['c_text'].apply(stem_snow)
df['c_pos_text'] = df.Idea.apply(pos_clean)

###############################################
''' TF-IDF + UMAP clustering'''
###############################################
df3 = prep_tfidf(df['c_text_snow'])

umap_model = umap.UMAP(random_state = 42, n_neighbors=10, min_dist=0.5, n_components=2).fit_transform(df3)

df['umap_x'] = umap_model[:,0]
df['umap_y'] = umap_model[:,1]

kmeans_1 = KMeans(n_clusters = 2, init='k-means++').fit(df.loc[::,['umap_x','umap_y']])
df['topic_umap_km'] = kmeans_1.predict(df.loc[::,['umap_x','umap_y']])

top5()
topics = list(set(df.topic_umap_km));
centroids = pd.DataFrame(kmeans_1.cluster_centers_, columns=['x','y'])
max_d = []
for i in topics: max_d.append(np.max(metrics.pairwise_distances(df.loc[df.topic_umap_km == i,['umap_x']], df.loc[df.topic_umap_km == i,['umap_y']])))

centroids.loc[:,'max_d'] = [i * 10 for i in max_d]

# Set up data
fact = list(map(str,sorted(list((df.topic_umap_km.unique()))))) 
col_dict = dict(zip(fact, viridis(len(fact))))
x = []
for i in df.index: x.append(col_dict[str(df.loc[i,'topic_umap_km'])])
df.loc[:,'col_umap'] = x

source = ColumnDataSource(pd.DataFrame(dict(x = df.umap_x, y = df.umap_y, c = df.col_umap, d = df.topic_umap_km, i = df.Idea, t = df.topic_top5w)).sort_values(by=['d'], inplace = False))

# Set up plot
TOOLTIPS = [('Idea', '@i'), ('Topic', '@d'), ('Top5','@t')]

plot = figure(plot_height=900, plot_width=900, title="UMAP text clustering", tooltips=TOOLTIPS, tools="crosshair,pan,reset,save,wheel_zoom,hover", x_range=[-20,20], y_range=[-20,20])
plot.scatter('x', 'y', 
           fill_color='c',
           size = 12, 
           line_color = None,
           legend_field = 'd',
           source = source)

plot.legend.click_policy="hide"
plot.legend.location = "top_left"

# Set up widgets
text = TextInput(title="title", value='UMAP Text clustering')

umap1 = Slider(title="NNeighbours", value=10, start=0, end=100, step=10)
umap2 = Slider(title="Min Dist.", value=0.5, start=0, end=1, step=0.1)
clusters = Slider(title="Clusters", value=1, start=1, end=100, step=1)

# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new): # Update Function KMeans Clustering

    cc = clusters.value
    kmeans = KMeans(n_clusters = cc, init='k-means++').fit(df.loc[::,['umap_x','umap_y']])
    df.loc[::,'topic_umap_km'] = kmeans.predict(df.loc[::,['umap_x','umap_y']])
       
    fact = list(map(str,sorted(list((df.topic_umap_km.unique()))))) 
    col_dict = dict(zip(fact, viridis(len(fact))))
    
    x = []
    for i in df.index: x.append(col_dict[str(df.loc[i,'topic_umap_km'])])
    
    df.loc[::,'col_umap'] = x
    
    top5()
         
    x2 = df.umap_x
    y2 = df.umap_y
    c2 = df.col_umap
    d2 = df.topic_umap_km
   
    source.data = pd.DataFrame(dict(x=x2, y=y2, c=c2, d=d2, i = df.Idea, t = df.topic_top5w)).sort_values(by=['d'], inplace = False)

def update_data2(attrname, old, new): 

    # Get the current slider values
    n1 = umap1.value
    n2 = umap2.value
    
    umap_model = umap.UMAP(random_state = 42, n_neighbors = n1, min_dist = n2, n_components=2).fit_transform(df3)
    df['umap_x'] = umap_model[:,0]
    df['umap_y'] = umap_model[:,1]
    
    fact = list(map(str,sorted(list((df.topic_umap_km.unique()))))) 
    col_dict = dict(zip(fact, viridis(len(fact))))
    
    x = []
    for i in df.index: x.append(col_dict[str(df.loc[i,'topic_umap_km'])])
    
    x2 = df.umap_x
    y2 = df.umap_y
    c2 = x
    d2 = df['topic_umap_km']
   
    source.data = pd.DataFrame(dict(x=x2, y=y2, c=c2, d=d2, i = df.Idea, t = df.topic_top5w)).sort_values(by=['d'], inplace = False)


for w in [clusters]:
    w.on_change('value', update_data)
    
for w in [umap1, umap2]:
    w.on_change('value', update_data2)

# Set up layouts and add to document
inputs = column(text, umap1, umap2, clusters)

curdoc().add_root(row(inputs, plot, width=900))
curdoc().title = 'UMAP Text Clustering'