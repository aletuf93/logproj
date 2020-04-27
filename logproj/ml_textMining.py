# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import nltk as nl
from difflib import SequenceMatcher

# %% BEST MATCH STRING

def findBestMatchingString(inputTable,compareStringList,old_label_column,new_label_column='MATCHED_STRING', matchingTreshold = 0.6, printMatchingString=True):
    #la funzione cerca per ogni riga del dataframe inputTable nella colonna old_label_column, 
    # la stringa di bestmatch nella lista inputTable
    # il risultato e' salvato nella colonna new_label_column
    # l'accuratezza e' limitata ad un indice di matching specificato da matchingTreshold (e' bene non sia inferiore a 0.6)
    
    #verifico che la colonna new_label_column esista
    if new_label_column not in inputTable.columns:
        inputTable[new_label_column]=np.nan
    
    #force mapping destination
    
    D_notMapped = list(set(inputTable[inputTable[new_label_column].isna()][old_label_column]))
    
    
    
    for destinazione in D_notMapped:
        #destinazione=D_notMapped[0]
        
        conteggio = [SequenceMatcher(None, destinazione, stringa).ratio() for stringa in compareStringList]
        
        checkBestfound = [count>matchingTreshold for count in conteggio]
        if any(checkBestfound):
            bestMatch = compareStringList[np.argmax(conteggio)]
            if printMatchingString:
                print(f"notMapped: {destinazione}, bestMatch: {bestMatch}")
            inputTable[new_label_column].loc[inputTable[old_label_column]==destinazione] = bestMatch
    return inputTable

# %%  BAG OF WORDS MODEL

def getFrequencyKeyword(inputTable,minFrequency,weightColumn,maxLenTable=[]):
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
        inputTable=inputTable.sort_values(by=[weightColumn], ascending=False)
        inputTable=inputTable.reset_index(drop=True)
        
        if isinstance(maxLenTable,int):
            inputTable=inputTable.iloc[0:maxLenTable]
        
        for i in range(0,len(inputTable)): 
        
                descWords=set(inputTable.WORDTAG[i].split(';'))
                descWords.remove('')
                weight=inputTable[weightColumn][i]
                for word in descWords:        
                    dictionary=dictionary.append(pd.DataFrame([[word,weight/len(descWords)]],columns=['word','frequency']))
        
        dictionary=dictionary.groupby(['word'])['frequency'].sum().reset_index()
        if len(dictionary)>0:
            #dictionary=pd.DataFrame.from_dict(words_except_stop_dist,orient='index',columns=['frequency'])
            dictionary.index=dictionary.word
            threshold=int((minFrequency/100)*len(dictionary))
            
            dictionary=dictionary.sort_values(by = ['frequency'], ascending=False)
            dictionary=dictionary.iloc[0:threshold]
            dictionary=dictionary.sort_values('frequency',ascending=False)
    return dictionary

# %%
def getDescriptionKeywords(D_SKU):

    #D_SKU restituisce la tabella di input con in agiunta una colonna che identifica le parole chiave per ogni record


    #dictionary restituisce il dataframe con le parole "dizionario" ogni parola compare una sola volta con la
    #relativa frequenza superiore a minFrequency
    #words_except_stop_dist restituisce tutte le parole con frequenza pronta per la generazione di un wordcloud


    txt = D_SKU.DESCRIPTION.str.lower().str.cat(sep=' ')

    wordToClean=['?','^','!',',','*','\'','/','\\','(',')',':','.','+',';','_','-','0','1','2','3','4','5','6','7','8','9','+' ]

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
        descWords=D_SKU['cleanDescription'].iloc[i]
        descWords=str(descWords)
        
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
                
            #D_SKU.at[i,'WORDTAG']=salva
            D_SKU['WORDTAG'].iloc[i]=salva

    

    return D_SKU
