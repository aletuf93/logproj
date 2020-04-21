# -*- coding: utf-8 -*

#import py_compile
#py_compile.compile('ZO_ML_unsupervisedModels.py')

import pandas as pd  
import numpy as np
import os




#Importa pacchetti grafici
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

#Import pacchetti clustering
from sklearn import cluster, datasets
from scipy.spatial.distance import pdist, jaccard,squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.mixture import GaussianMixture

#Import pacchetti statistici
from sklearn.preprocessing import MinMaxScaler

#Import pacchetti proprietari
#from ZO_ML_apriori import apriori  
#from ZO_ML_association_rules import association_rules
#from ZO_ML_machineLearning import dummyColumns



'''
def AprioriAlgorithm(dataFrame,minimo_support):
    supportDf=apriori(dataFrame,min_support=minimo_support, use_colnames=True)
    results=association_rules(supportDf)
    return results
'''
def groupVariableKMean(K_MAX_Kmeans,X,D_ITEM_clean,dirResults,connection,caseStudy,saveToDB=False):
    
    #crea una cartella per i risultati dei vessel
    dirResults = os.path.join(dirResults, 'Clustering_Kmean')
    try:
        os.mkdir(dirResults)
    except OSError:
        print('Cartella già esistente')
    
    results=pd.DataFrame(D_ITEM_clean.itemcode)
    for i in range(2,K_MAX_Kmeans+1):
        km = cluster.KMeans(n_clusters=i).fit(X)
        results[str(i)]=pd.DataFrame(km.labels_)
        fig1=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(X[:, 0], X[:, 1], c=km.labels_,cmap='plasma')
        plt.title("K="+str(i)+", J=%.2f" % km.inertia_)
        fig1.savefig(dirResults+'\\Kmeans'+str(i)+'.png')
        plt.close('all')
          
    return results



def GroupingVariableHierarchical(metodoGrouping,K_fixed_SLINK,X,D_ITEM_clean,dirResults,connection,caseStudy,saveToDB=False):
    
    #crea una cartella per i risultati
    dirResults = os.path.join(dirResults, 'Clustering_Hierarchical')
    try:
        os.mkdir(dirResults)
    except OSError:
        print('Cartella già esistente')
    
    results=pd.DataFrame(D_ITEM_clean.itemcode)
    for i in range(2,K_fixed_SLINK+1):
        fig1=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        hierCl = cluster.AgglomerativeClustering(n_clusters=i, linkage=metodoGrouping).fit(X)
        results[str(i)]=pd.DataFrame(hierCl.labels_)
        plt.scatter(X[:, 0], X[:, 1], c=hierCl.labels_,cmap='plasma')
        plt.title("K="+str(i))
        fig1.savefig(dirResults+'\\Hier_'+str(metodoGrouping)+'_'+str(i)+'.png')
        plt.close('all')
    return results
        
        
def HierarchicalClusterJaccard(D_table,X,targetColumn,itemcode,groupingMethod,NCluster,dirResults):
    
    #la itemcode è la colonna degli oggetti di cui si vuole costruire la similarità
    #la targetColumn è la colonna con valori (eventualmente) separati da virgola su cui costruire il jaccard
    #X definisce i punti sul piano per il grafico
    
    #crea una cartella per i risultati dei vessel
    dirResults = os.path.join(dirResults, 'Clustering_Hierarchical_'+str(targetColumn))
    try:
        os.mkdir(dirResults)
    except OSError:
        print('Cartella già esistente')
    
    D_Sim=D_table[targetColumn].str.get_dummies(sep=';')
    for j in D_Sim.columns : D_Sim[j]= D_Sim[j].astype(bool)
      
    
    Y = pdist(D_Sim, 'jaccard')
    Y[np.isnan(Y)]=0
    z = linkage(Y, method=groupingMethod)
    
    results=pd.DataFrame(D_table[itemcode])
    for k in range(1,NCluster+1):
        fig1=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        cutree = cut_tree(z, n_clusters=k)
        results[str(k)]=pd.DataFrame(cutree)
        plt.scatter(X[:, 0], X[:, 1], c=results[str(k)],cmap='plasma')
        plt.title("K="+str(k))
        fig1.savefig(dirResults+'\\Hier_'+str(groupingMethod)+'_'+str(k)+'.png')
        plt.close('all')

    return results       
                
    

def GroupingVariableGMM(K_fixed_GMM,X,D_ITEM_clean,dirResults,connection,caseStudy,saveToDB=False):
    
     #crea una cartella per i risultati dei vessel
    dirResults = os.path.join(dirResults, 'Clustering_GaussianMixture')
    try:
        os.mkdir(dirResults)
    except OSError:
        print('Cartella già esistente')
    results=pd.DataFrame(D_ITEM_clean.itemcode)
    for j in range(2,K_fixed_GMM+1):
        gmm = GaussianMixture(n_components=j, covariance_type='full').fit(X)
        results[str(j)]=pd.DataFrame(gmm.predict(X))
        
        #Rappresento i cluster ottenuti
        colors = sns.color_palette('plasma',max(results[str(j)])+1)
        fig1=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(X[:, 0], X[:, 1], c=[colors[lab] for lab in gmm.predict(X)])
        for i in range(gmm.covariances_.shape[0]):
            plot_cov_ellipse(cov=gmm.covariances_[i, :], pos=gmm.means_[i, :], facecolor='none', linewidth=2, edgecolor='black')
            plt.scatter(gmm.means_[i, 0], gmm.means_[i, 1], edgecolor=colors[i], marker="o", s=100, facecolor="w", linewidth=2)
        plt.title("K="+str(j))
        fig1.savefig(dirResults+'\GMM'+str(j)+'.png')
        plt.close('all')
    return results
        
       

def HierarchicalClusteringDendrogram(X,metodoGrouping,medotoDistanze,dirResults):
    
    #crea una cartella per i risultati dei vessel
    dirResults = os.path.join(dirResults, 'Clustering_Hierarchical')
    try:
        os.mkdir(dirResults)
    except OSError:
        print('Cartella già esistente')
    
    fig1=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    res = pdist(X, medotoDistanze)
    
    #costruisco linkage
    Z=linkage(res, method=metodoGrouping, metric=medotoDistanze)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('item')
    plt.ylabel('similarity')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    fig1.savefig(dirResults+'\Dendrogram_'+str(metodoGrouping)+'.png')
    plt.close('all')
    return True

#Funzione per disegnare ellissi nei modelli gaussiani


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def capacitatedClustering(D,simMin,dem,capacity):
    #D is a simmetric distance matrix
    #simMin minimum similarity value to aggregate two points
    #array of demand for each point
    #capacity = fixed capacity for each cluster
    
    method='single'
    
    
    select=len(D)
      
    
    
    #Parto da matrice delle distanze
    M=squareform(pdist(D))
    
    #scalo ad una matrice di prossimità (0,1. 1 sulla diagonale)
    scaler = MinMaxScaler()
    scaler.fit(M)
    M=scaler.transform(M)
    M=1-M
    
    #metto degli zero sulla diagonale per non clusterizzare un cluster con se stesso
    np.fill_diagonal(M,0)
    #ciclo di clustering
    progressivoCluster=0
    capCluster=np.zeros(select)
    capSatura=False
    
    while not(capSatura):
        progressivoCluster=progressivoCluster+1
        
        #faccio un ranking di tutti i punti
        simOrdered=np.unique(np.reshape(M,[M.size,1]))
        simOrdered=np.sort(simOrdered)
        simOrdered=simOrdered[simOrdered>=simMin]
        simOrdered=simOrdered[::-1] #ordino dal più grande al più piccolo
        
        if(len(simOrdered)==0): #se non ho candidati da clusterizzare ho finito
            capSatura=True
        
        trovato=False
        
        
        while ((not(trovato))&(not(capSatura))): #continuo a ciclare finché non ho capacità satura e finché non ho trovato il successivo nodo da aggregare
            
            for gg in range(0,len(simOrdered)): #così li considero tutti, compreso il primo che è un 1
                if((not(trovato))&(not(capSatura))):
                    simValue=simOrdered[gg] #scorro il ranking dei valori di similarità
                    inc=np.where(M==simValue) #identifico tutte le righe e colonne che hanno quel valore di similarità (a parimerito)
                    
                    #scorro i parimerito finché non ne trovo uno buono
                    for jj in range(0,len(inc[0])):
                        if((not(trovato))&(not(capSatura))):
                            max_id_row=inc[0][jj] #riga nodo candidato =>nodo 1 candidato all'aggregazione
                            max_id_column=inc[1][jj] #colonna candidato =>nodo 2 candidato all'aggregazione
                            
                            #verifico le condizioni di capacità
                            
                            
                            ############Identifico appartenenza al cluster del primo nodo candidato
                            #cerco nella matrica capCluster a quale cluster il nodo 1 è assegnato (se è 0 non è mai stato assegnato)
                            currentId1=capCluster[max_id_row]
                            if(not(currentId1==0)): #se è già stato assegnato in precedenza, eredito la capacità di tutti i nodi assegnati in quel cluster
                                currentId1=capCluster==currentId1
                            else: #altrimenti è un vettore di zeri con un uno in posizione del nodo (ancora mai assegnato)
                                currentId1=np.zeros(len(capCluster))
                                currentId1[max_id_row]=1
                                
                            ############Identifico appartenenza al cluster del secondo nodo candidato
                            currentId2=capCluster[max_id_column]
                            if(not(currentId2==0)): #se è già stato assegnato in precedenza, eredito la capacità di tutti i nodi assegnati in quel cluster
                                currentId2=capCluster==currentId2
                            else: #altrimenti è un vettore di zeri con un uno in posizione del nodo (ancora mai assegnato)
                                currentId2=np.zeros(len(capCluster))
                                currentId2[max_id_column]=1
                        
                        totalCapacity=currentId1+currentId2
                        totalCapacity=sum(dem*totalCapacity) #aggrego la capacità dei nodi candidati all'aggregazione
                        if(totalCapacity<capacity): #se rispetto la capacità, ho trovato i nodi da aggregare
                            trovato=True
                            
                    if((gg==len(simOrdered)-1)&(not(trovato))): #se ho ciclato su tutti i valori di similarità papabili e non sono riuscito ad aggregare ulteriormente  ho finito
                        capSatura=True
    
     #se ho trovato cluster da aggregare aggiorno i valori di similarità e i cluster
        if (not(capSatura)):
        
            #idenifico cluster 1
            currentiId1=capCluster[max_id_row]
            if(currentiId1==0): #se non l'ho mai aggregato aggiorno solo lui
                capCluster[max_id_row]=progressivoCluster
            else:
                capCluster[capCluster==currentiId1]=progressivoCluster       
        
            #idenifico cluster 2
            currentiId2=capCluster[max_id_column]
            if(currentiId2==0): #se non l'ho mai aggregato aggiorno solo lui
                capCluster[max_id_column]=progressivoCluster
            else:
                capCluster[capCluster==currentiId2]=progressivoCluster   
                
            #aggiorno i loro valori di similarità in matrice
            
            for h in range(0,len(M)): #scorro tutte le righe della matrice
                
                #aggiorno tutti gli indici relativi all'indice di colonna
                if((h!=max_id_column) & (h!=max_id_row)): #tranne quelle sulla diagonale
                    if method=='single':
                        M[h,max_id_column]=min(M[h,max_id_column],M[h,max_id_row])
                    #elif method=='complete':
                    #    M[h,max_id_column]=max(M[h,max_id_column],M[h,max_id_row])
                    #elif method=='average':
                    #    M[h,max_id_column]=np.mean(M[h,max_id_column],M[h,max_id_row])
                    M[max_id_column,h]=M[h,max_id_column] #rendo la matrice simmetrica   
                
                #aggiorno tutti gli indici relativi all'indice di riga
                if((h!=max_id_row) & (h!=max_id_column)):
                    if method=='single':
                        M[h,max_id_row]=min(M[h,max_id_row],M[h,max_id_column])
                    #elif method=='complete':
                    #    M[h,max_id_row]=max(M[h,max_id_row],M[h,max_id_column])
                    #elif method=='average':
                    #    M[h,max_id_row]=np.mean(M[h,max_id_row],M[h,max_id_column])
                    M[max_id_row,h]=M[h,max_id_row]
        
            #Azzero le similarità dei nodi che ho appena aggregato così che non vengano più pescati singolarmente
            M[max_id_row,max_id_column]=0
            M[max_id_column,max_id_row]=0
            
            #stampo i nodi appena clusterizzati
            #print(max_id_row,max_id_column)
    
    #se quando ho finito ho ancora dei nodi sul cluster zero, li assegno a diversi cluster       
    for jj in range(0,len(capCluster)):
        if(capCluster[jj]==0):
           capCluster[jj]=progressivoCluster
           progressivoCluster=progressivoCluster+1
    return progressivoCluster