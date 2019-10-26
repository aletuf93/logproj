# -*- coding: utf-8 -*

#import py_compile
#py_compile.compile('ZO_ML_unsupervisedModels.py')

import pandas as pd
import numpy as np
import os
import itertools
import time





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
from sklearn.manifold import TSNE



########### NEW  CLUSTERING METHODS ###########################################

def KMeansClustering(K,X,D_observations):
    #METODO KMEANS
    #costruisce un dataframe clusterizzando le N osservazioni nel vettore D_observation
    #in base ai valori N+M nel dataframe X
    #in K cluster



    results=pd.DataFrame(D_observations)
    km = cluster.KMeans(n_clusters=K).fit(X)
    results['CLUSTER']=pd.DataFrame(km.labels_)


    return results


def GMMClustering(K,X,D_observations):
    #METODO GAUSSIAN MIXTURE MODEL
    #costruisce un dataframe clusterizzando le N osservazioni nel vettore D_observation
    #in base ai valori N+M nel dataframe X
    #in K cluster

    results=pd.DataFrame(D_observations)
    gmm = GaussianMixture(n_components=K, covariance_type='full').fit(X)
    results['CLUSTER']=pd.DataFrame(gmm.predict(X))
    return results


def HierarchicalClustering(K,X,metodoGrouping,metodoDistanze,D_observations):
    #METODO CLUSTERING GERARCHICO

    #costruisce un dataframe clusterizzando le N osservazioni nel vettore D_observation
    #in base ai valori N+M nel dataframe X

    #metodoDistanze stabilisce il modo con cui calcolare la distanza fra le osservazioni
    #tipicamente jaccard, euclidea o cityblock

    #metodoGrouping stabilisce la logica di convergenza dei nodi (CLINK, SLINK, UPGMA)


    #crea una cartella per i risultati dei vessel
    results=pd.DataFrame(D_observations)

    #costruisco linkage
    hierCl = cluster.AgglomerativeClustering(n_clusters=K, affinity=metodoDistanze, linkage=metodoGrouping).fit(X)
    results['CLUSTER']=pd.DataFrame(hierCl.labels_)

    return results

########## OLD CLUSTERING METHODS #############################################

def AprioriAlgorithm(dataFrame,minimo_support):
    supportDf=apriori(dataFrame,min_support=minimo_support, use_colnames=True)
    results=association_rules(supportDf)
    return results

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


def capacitatedtSNEClustering(X,dem,capacity):
    #greedy capacitated clustering based on sorting based on a t-SNE
    #projection on a single dimension and filling the bins
    
    #D is the matrix with the coordinates of the points
    #dem is the matric containing the definition and the demand of each point
    #capacity is the maximum capacity static for each bin
    
    
    X_tsne = TSNE(n_components=1).fit_transform(X)
    res=dem.join(pd.DataFrame(X_tsne,columns=['TSNE']))
    res=res.sort_values(by='TSNE')
    res=res.reset_index(drop=True)
    res_cluster=[]
    currentCluster=1
    availableCapacity=capacity
    for i in range(0,len(res)):
        if availableCapacity>=res['fab_sec'][i]: #se ci sta lo assegno
            availableCapacity=availableCapacity-res['fab_sec'][i] #update capacity
            res_cluster.append(currentCluster) #assign point to cluster
        else: #se non ci sta apro un nuovo cluster
            availableCapacity=capacity-res['fab_sec'][i] #restore capacity
            currentCluster=currentCluster+1 #open new cluster
            res_cluster.append(currentCluster) #assign point to cluster
    res['CLUSTER']=res_cluster 
    return res
      
def capacitatedClustering(D,method,dem,capacity,stampa):
    #D is a matrix with observations
    #simMin minimum similarity value to aggregate two points
    #array of demand for each point
    #capacity = fixed capacity for each cluster
    #dem is the demand of each node

    '''
    esempio
    D=[[0,1],[0,0],[1,0],[1,1],[0.5,0.5]]
    capacity=3
    dem=[1,2,3,1,1]
    stampa=True
    method='average'
    clusters = capacitatedClustering(D,method,dem,capacity,stampa)
    '''
    #misuro tempo di esecuzione
    time_perf=[0]
    t = time.time()



    select=len(D)

    #Parto da matrice delle distanze
    M=squareform(pdist(D))

    #scalo ad una matrice di prossimità
    # 0.0001: (min similarità)
    # 1: max similarità
    # con 0 sulla diagonale
    # gli 0 non vengono presi in esame per l'aggregazione
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(M)
    M=scaler.transform(M)
    M=1-M
    M = np.maximum( M, M.transpose() ) #rendo simmetrica la matrice
    #metto degli zero sulla diagonale per non clusterizzare un cluster con se stesso
    np.fill_diagonal(M,-1)
    #ciclo di clustering
    progressivoCluster=0 #assegna codici ai cluster
    aggregazione=0 #conta il numero di aggregazioni
    capCluster=np.zeros(select)
    capSatura=False

    if stampa==True:
        print(f"Capacità massima: {capacity}")
        print("Matrice della domanda")
        print(dem)
        print(f"Metodo di clustering: {method} linkage")
        print("============================")
        print("Matrice di similarità iniziale")
        print(M)


    while not(capSatura):
        progressivoCluster=progressivoCluster+1 #apro un nuovo cluster

        #faccio un ranking di tutti i punti
        simOrdered=np.unique(np.reshape(M,[M.size,1]))
        simOrdered=np.sort(simOrdered)
        simOrdered=simOrdered[simOrdered>=0] #rimuovo tutti gli 0 (o sono sulla diagonale o già clusterizzati)
        #simOrdered=simOrdered[simOrdered>=simMin]
        simOrdered=simOrdered[::-1] #ordino dal più grande al più piccolo
        
        #numero massimo di opzioni da esaminare
        #maxLength=min(1000,len(simOrdered)) #ad ogni iterazione considero al massimo 1000 valori di similarità
        #simOrdered=simOrdered[0:maxLength]
        if(len(simOrdered)==0): #se non ho candidati da clusterizzare ho finito
            capSatura=True

        trovato=False
        #print(f"Valori di similarità da analizzare:{len(simOrdered)}")
        #questo ciclo while satura la capacità di un cluster
        #while ((not(trovato))&(not(capSatura))): #continuo a ciclare finché non ho capacità satura e finché non ho trovato il successivo nodo da aggregare
            #questo ciclo scorre tutti i valori di similarità "papabili" -cioè > simMin
        
        
        for gg in range(0,len(simOrdered)): #così li considero tutti, compreso il primo che è un 1
            t_inizio_ciclo = time.time()
            if (trovato | capSatura):
                break
            else:
                    simValue=simOrdered[gg] #scorro il ranking dei valori di similarità
                    inc=np.where(M==simValue) #identifico tutte le righe e colonne che hanno quel valore di similarità (a parimerito)
                    points_couples=list(zip(inc[0], inc[1])) #creo coppie di punti
                    couples=list(set(tuple(sorted(l)) for l in points_couples)) #rimuovo duplicati sulle permutazioni (1-2 == 2-1)
                    #scorro i parimerito finché non ne trovo uno buono
                    #print(f"Coppie da analizzare:{len(couples)}")
                    for jj in couples:
                        t_elapsed = time.time() - t_inizio_ciclo
                        if((not(trovato)) & (t_elapsed<=60)): #scorro tutti i valori papabili finché ne trovo uno buono o per un massimo di un minuto
                            max_id_row=jj[0]#riga nodo candidato =>nodo 1 candidato all'aggregazione
                            max_id_column=jj[1] #colonna candidato =>nodo 2 candidato all'aggregazione

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
                            if(totalCapacity<=capacity): #se rispetto la capacità, ho trovato i nodi da aggregare
                                trovato=True

                    if((gg==len(simOrdered)-1)&(not(trovato))): #se ho ciclato su tutti i valori di similarità papabili e non sono riuscito ad aggregare ulteriormente  ho finito
                        capSatura=True
     #dal ciclo sopra mi porto a casa max_id_row e max_id_column dei due nodi da aggregare
     #se ho trovato cluster da aggregare aggiorno i valori di similarità e i cluster
        if (not(capSatura)):

            #idenifico cluster 1
            currentiId1=capCluster[max_id_row] #scrivo su capCluster l'assegnamento di 1
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
                    elif method=='complete':
                        M[h,max_id_column]=max(M[h,max_id_column],M[h,max_id_row])
                    elif method=='average':
                        M[h,max_id_column]=np.mean([M[h,max_id_column],M[h,max_id_row]])
                    M[max_id_column,h]=M[h,max_id_column] #mantengo la matrice simmetrica

                #aggiorno tutti gli indici relativi all'indice di riga
                if((h!=max_id_row) & (h!=max_id_column)):
                    if method=='single':
                        M[h,max_id_row]=min(M[h,max_id_row],M[h,max_id_column])
                    elif method=='complete':
                        M[h,max_id_row]=max(M[h,max_id_row],M[h,max_id_column])
                    elif method=='average':
                        M[h,max_id_row]=np.mean([M[h,max_id_row],M[h,max_id_column]])
                    M[max_id_row,h]=M[h,max_id_row] #mantengo la matrice simmetrica

            #Azzero le similarità dei nodi che ho appena aggregato così che non vengano più pescati singolarmente
            #M[max_id_row,max_id_column]=-1
            #M[max_id_column,max_id_row]=-1
            clusterCreato = np.where(capCluster==progressivoCluster) # identifico tutti i nodi dello stesso cluster
            for node1, node2 in list(itertools.combinations(clusterCreato[0].tolist(), 2)):
                #print(str(node1)+"-"+str(node2))
                if M[node1,node2]!=-1: #scrivo -1 solo se già non l'ho scritto
                    M[node1,node2]=-1
                    M[node2,node1]=-1
            # non solo loro due, ma anche tutti uelli già dentro allo stesso cluster!!!!!

            #stampo i nodi appena clusterizzati
            time_perf.append(time.time() - t)
            t = time.time()
            aggregazione=aggregazione+1
            print(f"Iterazione: {aggregazione}/{len(dem)-1}")
            if stampa==True:
                print("============================")
                print(f"Aggrego nodi: {max_id_row} e {max_id_column}")
                print("Matrice di similarità aggiornata")
                print(M)


    #se quando ho finito ho ancora dei nodi sul cluster zero, li assegno a diversi cluster
    for jj in range(0,len(capCluster)):
        if(capCluster[jj]==0):
           capCluster[jj]=progressivoCluster
           progressivoCluster=progressivoCluster+1

    if stampa==True:
        print("osservazioni aggregate nei cluster")
        print(capCluster)
    return capCluster,time_perf
