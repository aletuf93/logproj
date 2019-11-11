# -*- coding: utf-8 -*-

#import py_compile
#py_compile.compile('ZO_ML_diagnosticAnalysis.py')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import networkx as nx
import numpy as np
import nltk as nl



def scale_range (series, minimo, massimo):
    series += -(np.min(series))
    series /= np.float(np.max(series)) / (massimo - minimo)
    series += minimo
    return series


def paretoChart(df, barVariable, paretoVariable):

    #barVariable è la variabile di conteggio (istogramma) (itemcode)
    #paretoVariable è la variabile numerica (popularity)

    df = df.sort_values(by=paretoVariable,ascending=False)
    df["cumpercentage"] = df[paretoVariable].cumsum()/df[paretoVariable].sum()*100


    fig, ax = plt.subplots(figsize=(20,10))

    #asse principale
    ax.bar(np.linspace(0, 100, num=len(df)), df[paretoVariable], color="C0", width=0.5)
    #ax.bar(df[barVariable], df[paretoVariable], color="C0")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")

    #asse secondario
    ax2 = ax.twinx()
    ax2.plot(np.linspace(0, 100, num=len(df)), df["cumpercentage"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params(axis="y", colors="C1")
    plt.title('Pareto '+str(barVariable))
    plt.xlabel(str(barVariable))
    plt.ylabel('Percentage '+str(paretoVariable))
    plt.ylim([0 , 110])
    return fig

def plotGraph(df,edgeFrom,edgeTo,distance,weight,title,arcLabel=True):
    G=nx.from_pandas_edgelist(df, edgeFrom, edgeTo,edge_attr=True,create_using=nx.DiGraph())


    edges = G.edges()
    weights = [G[u][v][weight] for u,v in G.edges]
    labels = nx.get_edge_attributes(G,weight)

    pos = nx.layout.spring_layout(G,weight =distance)

    #node_sizes = weights
    #M = G.number_of_edges()
    edge_colors = weights
    edge_width=scale_range(np.float_(weights),1,10)
    #edge_alphas = weights

    fig1=plt.figure(figsize=(20,10))
    plt.title('Flow analysis '+str(title))
    nx.draw(G,pos,node_size=0,edge_color='white',with_labels = True)
    #nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='Orange')
    edges = nx.draw_networkx_edges(G, pos, width=edge_width, arrowstyle='->',
                                    edge_color=edge_colors,
                                   edge_cmap=plt.cm.Wistia)
    if arcLabel:
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    # set alpha value for each edge
    #for k in range(M):
    #    edges[k].set_alpha(edge_alphas[k]/max(edge_alphas))

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Wistia)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    return fig1

def getFrequencyKeyword(inputTable,minFrequency):
    #take as input a dataframe with:
    # - a column WORDTAG having keywords separated by ;
    # - a column CONTEGGIO with the weight of each row of the dataframe
    #return a dataframe with a single word and its frequency among all the table
    dictionary=pd.DataFrame(columns=['word','frequency'])
    
    #filter the table
    filter1=inputTable.WORDTAG.isnull()
    filter2=inputTable.WORDTAG == np.nan
    filter=filter1 | filter2
    inputTable=inputTable[~filter]   
    
    if len(inputTable)>0:
        inputTable=inputTable.reset_index(drop=True)
        
        for i in range(0,len(inputTable)): 
        
                descWords=set(inputTable.WORDTAG[i].split(';'))
                descWords.remove('')
                weight=inputTable['CONTEGGIO'][i]
                for word in descWords:        
                    dictionary=dictionary.append(pd.DataFrame([[word,weight/len(descWords)]],columns=['word','frequency']))
        
        dictionary=dictionary.groupby(['word'])['frequency'].sum().reset_index()
        if len(dictionary)>0:
            #dictionary=pd.DataFrame.from_dict(words_except_stop_dist,orient='index',columns=['frequency'])
            dictionary.index=dictionary.word
            threshold=int((1-minFrequency/100)*max(dictionary.frequency))
            dictionary=dictionary[dictionary.frequency>threshold]
            dictionary=dictionary.sort_values('frequency',ascending=False)
    return dictionary

def getDescriptionKeywords(D_SKU):

    #D_SKU restituisce la tabella di input con in agiunta una colonna che identifica le parole chiave per ogni record


    #dictionary restituisce il dataframe con le parole "dizionario" ogni parola compare una sola volta con la
    #relativa frequenza superiore a minFrequency
    #words_except_stop_dist restituisce tutte le parole con frequenza pronta per la generazione di un wordcloud


    txt = D_SKU.DESCRIPTION.str.lower().str.cat(sep=' ')

    wordToClean=['?','^','!',',','*','\'','/','\\','(',')',':','.',';','_','-','0','1','2','3','4','5','6','7','8','9','+' ]

    for w in wordToClean:
        txt=txt.replace(w, ' ')

    #divide in singole parole in base a spazi e punteggiature
    words = nl.tokenize.word_tokenize(txt)

    #importo le stopwords
    stopwords = nl.corpus.stopwords.words('italian')

    #elimino le parole troppo corte e quelle fra le stopwords
    wc_vocabulary=(w for w in words if (len(w)>3) & (w not in stopwords))
    words_except_stop_dist=nl.FreqDist(wc_vocabulary)
    #word_dist = nl.FreqDist(words_except_stop_dist)
    #fig1=plt.figure(figsize=(20,10))
    #word_dist.plot(50,cumulative=False,color='orange')
    #plt.title('Bag of Words')
    #fig1.savefig(dirResults+'\\'+'descriptionsFrequency.png')

    bagOfWords=pd.DataFrame.from_dict(words_except_stop_dist,orient='index',columns=['frequency'])
    #bagOfWords=bagOfWords[bagOfWords.frequency>minFrequency]

    D_SKU['cleanDescription']=D_SKU.DESCRIPTION.str.lower()
    for w in wordToClean:
        D_SKU['cleanDescription']=D_SKU['cleanDescription'].str.replace(w, ' ')

    D_SKU['WORDTAG']='NullPredictor'

    #output with frequency for each word
    
    for i in range(0,len(D_SKU)):
        descWords=D_SKU['cleanDescription'][i]
        
        if(not(descWords is None)):
            descWords=set(descWords.split(' '))
            keyWords=descWords.intersection(set(bagOfWords.index.values))
            salva=''
            num_caratteriDisponibili=149
            for s in keyWords:
                if(num_caratteriDisponibili>len(s)+1): #Ho solo 150 caratteri a database
                    num_caratteriDisponibili=num_caratteriDisponibili-len(s)-1
                    salva=salva+';'+(str(s))
                    salva=salva[0:]
                
            D_SKU.at[i,'WORDTAG']=salva

    

    return D_SKU
