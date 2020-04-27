# graph utility for warehouse optimisation

#%% import packages
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
#import missingno as msgn
from networkx.readwrite import json_graph
import json

#%% import packages from other folders

import database.mongo_queries as qq
import database.mongo_loginManager as mdb
import logproj.ml_graphs as dg
#from logproj.ml_machine_learning import cleanUsingIQR

#%% define graph G(V,E) and save to mongoDB

def createAndSaveGraph(caseStudy,nodecode):
    GG, D_res, D_layout = defineWHgraph(caseStudy, nodecode, [], [], [], draw=False, arcLabel=False, nodeLabel=False, trafficGraph=False)
    #print(f"{dt.datetime.now()}====================Graph created")
    
    #convert the graph to JSON object
    #import json
    json_graph_converted = json_graph.node_link_data(GG)
    json_graph_converted['locationMatches']=json.dumps(D_res)
    json_graph_converted['nodecode']=str(nodecode)
    
    #setto la connessione a MongoDB
    db, _ = mdb.setConnectionPymongo(str(caseStudy))
    graph_collection=db['graph_wh'] #seleziono le collezione corretta
    graph_collection.delete_many({}) # rimuovo eventuali grafi salvati precedentemente
    okInsert = graph_collection.insert_one(json_graph_converted) #inserisco il nuovo grafo
    return okInsert

def importGraph(caseStudy,nodecode):
    
    #setto la connessione a MongoDB
    db, _ = mdb.setConnectionPymongo(str(caseStudy))
    graph_collection=db['graph_wh'] #seleziono le collezione corretta
    
    #importo il grafo
    data=graph_collection.find_one({"nodecode":str(nodecode)})
    
    #data2 = json.loads(s1)
    G=nx.json_graph.node_link_graph(data)
    D_res = data['locationMatches']
    D_res=json.loads(D_res)
    D_res = {int(float(k)):int(float(v)) for k,v in D_res.items()}
    return G, D_res

# %%
def defineCoordinatesFromRackBayLevel(D_layout, aisleX=5.0, bayY=0.9):
    # definisce le coordinate x e y per ogni location in base al
    # numero di corsia (rack)
    # numero di campata (bay)
    # da utilizzare quando le coordinate cartesiane non vengono mappate
    # scrive in output sulle colonne loccodex e loccodey del dataframe D_layout

    print(f"Assuming aisle width of {aisleX} meters and bay width (pallet) of {bayY} meters")

    #identifico corsie
    D_layout['loccodex']=-1
    D_layout['loccodey']=-1
    allAisles=list(set(D_layout.rack))
    allAisles.sort()
    j=0
    #scorro tutte le corsie
    for x in allAisles:
        #assegno la coordinata x in base alla distanza fra i corridoi
        idx_x=D_layout.rack==x
        D_layout['loccodex'].loc[idx_x]=aisleX*j
        j=j+1

        #identifico tutte le campate in corsia
        allBays=list(set(D_layout['bay'].loc[idx_x]))
        i=0
        for y in allBays:
            #assegno la coordinata y in base al passo fra una campata e l'altra
            # per ipotesi tutte le corsie iniziano sul fronte
            idx_y=(D_layout.rack==x) & (D_layout.bay==y)
            D_layout['loccodey'].loc[idx_y]=bayY*i
            i=i+1
    return D_layout


# %%
def estimateMissingAislecoordX(D_layout,draw=False):
    
    #salvo dataset iniziale
    '''
    if draw:
        msgn.matrix(D_layout)
        plt.title("Initial Layout Data")
        plt.savefig("01InitialDataset.png")
    '''
    #stima i valori della coordinata della corsia quando non sono stati mappati (colonna aislecodex del dataframe D_layout)

    #####################################################
    #### sostituisco i nulli in loccodex e loccodey #####
    #####################################################

    #se ho l'indicazione dei rack
    if 'rack' in D_layout.columns:
            D_layout=D_layout.sort_values(['rack', 'bay'], ascending=[True, True])
            allRacks=list(set(D_layout.rack.dropna()))
            for rack in allRacks:

                D_rack=D_layout[D_layout.rack==rack]

                #provo a calcolarmi un valor medio della corsia
                avgXCoord=np.mean(D_rack.loccodex)
                if not(math.isnan(avgXCoord)): #se ho trovato un valore
                    D_rack['loccodex'].fillna(avgXCoord, inplace=True)

                else:# se ho tutti valori nulli cerco nell'intorno e interpolo
                    D_rack_null = D_layout[['rack','loccodex']].drop_duplicates()
                    D_rack_null=D_rack_null.sort_values('rack')
                    D_rack_null['loccodex'].fillna(method='backfill', inplace=True)
                    fillValue=float(D_rack_null[D_rack_null.rack==rack].loccodex)
                    # A questo punto sostituisco
                    D_rack['loccodex'].fillna(fillValue, inplace=True)


                #a questo punto setto i valori delle corsie in base a nearest neighbor
                D_rack['loccodey'].interpolate(method ='linear', limit_direction ='forward', inplace=True)

                #aggiorno D_layout
                D_layout.loc[D_rack.index] = D_rack
                
            #elimino eventuali nulli rimasti
            D_layout=D_layout.sort_values(by=['rack','bay'])
            print(f"====={len(D_layout[D_layout.loccodex.isnull()])} x coordinates have been randomly interpolated")
            D_layout['loccodex'].fillna(method='ffill', inplace=True) # riempie scorrendo in avanti se ci sono ulteriori nulli
            D_layout['loccodex'].fillna(method='bfill', inplace=True) # riempie scorrendo in avanti se ci sono ulteriori nulli
            
            
    else:
        print("No rack information")
           
    '''
    if draw:
        msgn.matrix(D_layout)
        plt.title("Fill LoccodeX and LoccodeY")
        plt.savefig("02FillXY.png")
    '''
    #####################################################
    ###### stimo coordinate delle corsie mancanti #######
    #####################################################

    # identifico le coordinate delle corsie (aislecodex) mappate
    D_givAisl=D_layout[D_layout['aislecodex'].notna()]
    D_givAisl=D_givAisl[['loccodex','aislecodex']]
    D_givAisl=D_givAisl.drop_duplicates()

    # identifico le coordinate delle corsie da mappare
    D_estAisl=D_layout[D_layout['loccodex'].notna()].loccodex
    allXcoords=list(set(D_estAisl))
    allXcoords.sort()

    #accoppio le coordinate, metto nella stessa corsia le piu' lontane
    dist=0
    for j in range(1,len(allXcoords)):
        dist=dist+np.abs(allXcoords[j]-allXcoords[j-1])
    if len(allXcoords)>1:
        avg_dist=dist/(len(allXcoords)-1)
    else:
        avg_dist=0

    #se la distanza e' maggiore alla media accoppio nella stessa corsia
    D_estAisl=pd.DataFrame(columns=D_givAisl.columns)
    j=0
    while j<len(allXcoords):
        if j < len(allXcoords)-1: #per ogni corsia eccetto l'ultima
            dist=np.abs(allXcoords[j+1]-allXcoords[j])
            if dist>=avg_dist: # se sono piu' lontane della media affacciano sulla stessa corsia (vale anche in caso di parita' cos' da considerare il caso in cui siano equidistanziate)
                aisle=min(allXcoords[j+1],allXcoords[j]) + dist/2
                D_estAisl=D_estAisl.append(pd.DataFrame([[allXcoords[j],aisle]],columns=D_estAisl.columns))
                D_estAisl=D_estAisl.append(pd.DataFrame([[allXcoords[j+1],aisle]],columns=D_estAisl.columns))
                j=j+2 # ho accopiato due, salto di due
            else: #altrimenti fa corsia da sola
                D_estAisl=D_estAisl.append(pd.DataFrame([[allXcoords[j],allXcoords[j]]],columns=D_estAisl.columns))
                j=j+1 # ho accoppiato di una, salto di una
        elif j == len(allXcoords)-1: # se sono all'ultima corsia
            D_estAisl=D_estAisl.append(pd.DataFrame([[allXcoords[j],allXcoords[j]]],columns=D_estAisl.columns))
            j=j+1 # ho accoppiato di una, salto di una

    

    #plt.scatter(allXcoords, np.ones(len(allXcoords)))
    #plt.scatter(D_estAisl.loccodex, np.ones(len(allXcoords)))
    #plt.scatter(D_estAisl.aislecodex, np.ones(len(allXcoords)), c='r', marker='*', s=2)

    #  data cleaning

    #replace None with nan
    D_layout.replace(to_replace=[None], value=np.nan, inplace=True)
    #check null aisle values
    index = D_layout['aislecodex'].index[D_layout['aislecodex'].apply(np.isnan)]


    for rows in index:
        loccodex=D_layout.loc[rows].loccodex


        #if the value is known
        if loccodex in D_givAisl.loccodex:
            D_layout['aislecodex'].loc[rows]=float(D_givAisl[D_givAisl['loccodex']==loccodex].aislecodex)
        else:
            D_layout['aislecodex'].loc[rows]=float(D_estAisl[D_estAisl['loccodex']==loccodex].aislecodex)
    '''
    if draw:
        msgn.matrix(D_layout)
        plt.title("Fill aislecodeX")
        plt.savefig("03FillaislecodeX.png")
    '''
    #check if coordinates exist otherwise replace with rack/bay/level

    #remove rack/bay/level
    D_layout=D_layout.sort_values(by=['rack','bay'])
    D_layout=D_layout[['idlocation', 'aislecodex', 'loccodex', 'loccodey']]

    #interpolo eventuali coordinate y rimaste scoperte (ultima spiaggia)
    
    print(f"====={len(D_layout[D_layout.loccodey.isnull()])} y coordinates have been randomly interpolated")
    D_layout['loccodey'].interpolate(method ='linear', limit_direction ='forward', inplace=True)
    D_layout['loccodey'].fillna(method='ffill', inplace=True) # riempie scorrendo in avanti se ci sono ulteriori nulli
    D_layout['loccodey'].fillna(method='bfill', inplace=True) # riempie scorrendo in avanti se ci sono ulteriori nulli
    
    '''       
    if draw:
        msgn.matrix(D_layout)
        plt.title("Final dataset")
        plt.savefig("04Fill nan loccodey.png")
    
        plt.close('all')
    '''
    #remove null
    #D_layout=D_layout.dropna()
    
    #arrotondo le x al metro e le y al decimetro per ridurre errori nella mappatura
    D_layout['aislecodex']=np.round(D_layout['aislecodex'],0)
    D_layout['loccodey']=np.round(D_layout['loccodey'],0)
    
    return D_layout

# %%
def defineGraphNodes(D_layout, D_IO):
        #la funzione definisce la corrispondenza fra idlocation e nodi
        #vengono definite corrispondenze per locazioni fisiche e IO
        # (alle locazioni fittizzie sono gia' state assegnate le coordinate dell'IO)
        #la funzione restituisce una tabella D_nodes con le coordinate dei nodi
        #un dizionario D_res con la corrispondenza fra idlocation (key) e idnode (values)
        #un dataframe D_IO con le coordinate di input/output



        #  definisco tutti i nodi del grafo
        D_nodes=D_layout[['aislecodex','loccodey']].drop_duplicates().reset_index(drop=True)
        #plt.scatter(D_nodes.aislecodex, D_nodes.loccodey)

        #aggiungo corrispondenza fra D_layout e D_nodes
        D_layout['idNode']=None
        for index, node in D_nodes.iterrows():
            idx_node=(D_layout.aislecodex==node.aislecodex) & (D_layout.loccodey==node.loccodey)
            D_layout.idNode.loc[idx_node]=index

        #aggiungo i nodi di IO
        #redefine index of D_IO to avoid overlaps with D_nodes
        D_IO.index = np.arange(max(D_nodes.index.values)+1, max(D_nodes.index.values) + 1 +  len(D_IO))

        for index, node in D_IO.iterrows():
            idx_node=node.idlocation # prendo l'idlocation della fake
            temp = pd.DataFrame([[idx_node, node.loccodex, node.loccodex, node.loccodey, index]],columns=D_layout.columns)
            D_layout=D_layout.append(temp)



        D_res=D_layout[['idlocation','idNode']]
        D_res=D_res.drop_duplicates()
        #D_res.set_index('idlocation',drop=True)
        #D_res=D_res['idNode'].to_dict()

        D_res_dict = dict(zip(D_res.idlocation, D_res.idNode))

        return D_nodes, D_res_dict, D_IO



def addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, index_source, index_target):
        D_Aisle1=D_nodes[D_nodes.aislecodex==list_aisles[index_source]] #identifico le coordinate della prima corsia
        D_Aisle2=D_nodes[D_nodes.aislecodex==list_aisles[index_target]]  #identifico le coordinate della prima corsia
        
        #se mi trovo a collegare due corsie "tradizionali" (entrambe con piu' di una ubica)
        if (len(D_Aisle1)>1) & (len(D_Aisle2)>1):
                #identifico le due ubiche sul fondo
            node1_front_index=D_Aisle1['loccodey'].idxmax()
            node2_front_index=D_Aisle2['loccodey'].idxmax()
    
            #aggiungo l'arco
            #nodeFrom=D_Aisle1.index[node1_front_index]
            #nodeTo=D_Aisle2.index[node2_front_index]
            length=np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index]-D_Aisle2.aislecodex.loc[node2_front_index]),1)
            temp=pd.DataFrame([[node1_front_index,node2_front_index,length]],columns=columns_edgeTable)
            edgeTable=edgeTable.append(temp)
            #print([node1_front_index,node2_front_index])
    
    
            #identifico le due ubiche sul fronte
            node1_front_index=D_Aisle1['loccodey'].idxmin()
            node2_front_index=D_Aisle2['loccodey'].idxmin()
    
            #aggiungo l'arco
            #nodeFrom=D_Aisle1.index[node1_front_index]
            #nodeTo=D_Aisle2.index[node2_front_index]
            length=np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index]-D_Aisle2.aislecodex.loc[node2_front_index]),1)
            temp=pd.DataFrame([[node1_front_index,node2_front_index,length]],columns=columns_edgeTable)
            edgeTable=edgeTable.append(temp)
            
        else: #qui sto connettendo ubiche singole (ad esempio zone a terra)
            
            if len(D_Aisle1)>1: # se la prima e' una corsia tradizionale
                
                #identifico le due coordinate della prima corsia
                node1_back_index=D_Aisle1['loccodey'].idxmax()
                node1_front_index=D_Aisle1['loccodey'].idxmin()
                
                node2_front_index=D_Aisle2['loccodey'].idxmax() # restituisce l'indice dell'unica ubica
                
                #effettuo solo un collegamento alla piu' vicina (calcolo entrambe le distanze)
                length_back=np.round(np.abs(D_Aisle1.aislecodex.loc[node1_back_index]-D_Aisle2.aislecodex.loc[node2_front_index]) + np.abs(D_Aisle1.loccodey.loc[node1_back_index]-D_Aisle2.loccodey.loc[node2_front_index]),1)
                length_front=np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index]-D_Aisle2.aislecodex.loc[node2_front_index]) + np.abs(D_Aisle1.loccodey.loc[node1_front_index]-D_Aisle2.loccodey.loc[node2_front_index]),1)
                
                #se e' piu' corto sul front aggiungo solo l'arco davanti
                if length_front<=length_back:
                    temp=pd.DataFrame([[node1_front_index,node2_front_index,length_front]],columns=columns_edgeTable)
                    edgeTable=edgeTable.append(temp)
                else: # altrimenti quello dietro
                    temp=pd.DataFrame([[node1_back_index,node2_front_index,length_back]],columns=columns_edgeTable)
                    edgeTable=edgeTable.append(temp)
   
            else: # tutti gli altri casi (ubica-ubica oppure ubica-corsia tradizionale)
                
                #identifico la coordinata della prima corsia
                node1_front_index=D_Aisle1['loccodey'].idxmax()
                
                #identifico la/le coordinate della seconda
                node2_back_index=D_Aisle2['loccodey'].idxmax()
                node2_front_index=D_Aisle2['loccodey'].idxmin()
                
                #effettuo solo un collegamento alla piu' vicina (calcolo entrambe le distanze)
                length_back=np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index]-D_Aisle2.aislecodex.loc[node2_back_index]) + np.abs(D_Aisle1.loccodey.loc[node1_front_index]-D_Aisle2.loccodey.loc[node2_back_index]),1)
                length_front=np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index]-D_Aisle2.aislecodex.loc[node2_front_index]) + np.abs(D_Aisle1.loccodey.loc[node1_front_index]-D_Aisle2.loccodey.loc[node2_front_index]),1)
                
                #se e' piu' corto sul front aggiungo solo l'arco davanti
                if length_front<=length_back:
                    temp=pd.DataFrame([[node1_front_index,node2_front_index,length_front]],columns=columns_edgeTable)
                    edgeTable=edgeTable.append(temp)
                else: # altrimenti quello dietro
                    temp=pd.DataFrame([[node1_front_index,node2_back_index,length_back]],columns=columns_edgeTable)
                    edgeTable=edgeTable.append(temp)
            
        return edgeTable


# %%
# define the edgeTable
def defineEdgeTable(D_nodes, D_IO):
    #la funzione definisce un dataframe di archi composto dalle colonne nodefrom, nodeto, length

    # elimino temporaneamente le coordinate relative all'IO su cui sono state mappate anche le fittizie
    D_fakes=pd.DataFrame(columns=D_nodes.columns)
    for index, row in D_IO.iterrows():
        loccodex=row.loccodex
        loccodey=row.loccodey
        #print(f"x:{loccodex},y:{loccodey}")
        #print(D_nodes[((D_nodes.aislecodex==loccodex) & (D_nodes.loccodey==loccodey))])
        D_fakes=D_fakes.append(D_nodes[((D_nodes.aislecodex==loccodex) & (D_nodes.loccodey==loccodey))])
        D_nodes=D_nodes[~((D_nodes.aislecodex==loccodex) & (D_nodes.loccodey==loccodey))]



    columns_edgeTable=['nodeFrom', 'nodeTo','length']
    edgeTable=pd.DataFrame(columns=columns_edgeTable)

    ######################################################
    ###### aggiungo archi verticali (corsie)##############
    ######################################################
    set_aisles=set(D_nodes.aislecodex) #identifico tutte le corsie
    for aisle in set_aisles:
        #aisle=list(set_aisles)[0]
        D_currentAisle=D_nodes[D_nodes.aislecodex==aisle] # filtro per corsia
        D_currentAisle=D_currentAisle.sort_values(by='loccodey') # ordino per campata

        #semplifico il grafo identificando tutte le location solo sulla coordinata y
        for i in range(1,len(D_currentAisle)): # identifico gli archi

            #identifico parametri di archi e loro attributi
            nodeFrom=D_currentAisle.index[i-1]
            nodeTo=D_currentAisle.index[i]
            length=np.round(np.abs(D_currentAisle.loccodey.iloc[i-1]-D_currentAisle.loccodey.iloc[i]),1)

            temp=pd.DataFrame([[nodeFrom,nodeTo,length]],columns=columns_edgeTable)
            edgeTable=edgeTable.append(temp)

    ######################################################
    ###### aggiungo archi trasversali ####################
    ######################################################
    
    list_aisles=list(set_aisles) # identifico le coordinate di ogni corsia
    list_aisles.sort() # ordino per coordinata
    for i in range(1,len(list_aisles)):
        #considero l'indice in cui mi trovo per creare un arco con la corsia vicina e quella successiva (2 corsie)
        if i==1: 
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i)
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i+1)
        elif i==len(list_aisles)-1:
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i)
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i-2)
        else:
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i)
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i-2)
            edgeTable=addtraversaledges(D_nodes,list_aisles,edgeTable,columns_edgeTable, i-1,i+1)
            
        
        
            



    ######################################################
    ###### aggiungo archi di IO e fittizie################
    ######################################################


    # find input
    D_in = D_IO[D_IO.inputloc==1]
    for idx in D_in.index:
        #identifico le coordinate
        loccodex=D_in.loccodex[idx]
        loccodey=D_in.loccodey[idx]

        #identifico il nodo piu' vicino
        distanceArray=np.abs(D_nodes.aislecodex-loccodex) +np.abs(D_nodes.loccodey-loccodey)
        idx_min=distanceArray.idxmin()
        length=min(distanceArray)

        #creo l'arco
        nodeFrom=idx
        nodeTo=idx_min
        temp=pd.DataFrame([[nodeFrom,nodeTo,length]],columns=columns_edgeTable)
        edgeTable=edgeTable.append(temp)

        #identifico eventuali fittizie mappate sulle stesse coordinate
        for idx_fake, row_fake in D_fakes.iterrows():
            #se ho una fittizzia mappata nella stessa locazione di IO
            if ((row_fake.aislecodex==loccodex) & (row_fake.loccodey==loccodey)):
                # aggiungo l'arco
                nodeFrom=idx_fake
                temp=pd.DataFrame([[nodeFrom,nodeTo,0]],columns=columns_edgeTable)
                edgeTable=edgeTable.append(temp)


    # find output
    D_out = D_IO[D_IO.outputloc==1]
    for idx in D_out.index:
        #identifico le coordinate
        loccodex=D_out.loccodex[idx]
        loccodey=D_out.loccodey[idx]

        #identifico il nodo piu' vicino
        distanceArray=np.abs(D_nodes.aislecodex-loccodex) + np.abs(D_nodes.loccodey-loccodey)
        idx_min=distanceArray.idxmin()
        length=min(distanceArray)

        #creo l'arco
        nodeFrom=idx
        nodeTo=idx_min
        temp=pd.DataFrame([[nodeFrom,nodeTo,length]],columns=columns_edgeTable)
        edgeTable=edgeTable.append(temp)

        #identifico eventuali fittizie mappate sulle stesse coordinate
        for idx_fake, row_fake in D_fakes.iterrows():
            #print(row_fake)
            #se ho una fittizzia mappata nella stessa locazione di IO
            if ((row_fake.aislecodex==loccodex) & (row_fake.loccodey==loccodey)):

                # aggiungo l'arco
                nodeFrom=idx_fake
                temp=pd.DataFrame([[nodeFrom,nodeTo,0]],columns=columns_edgeTable)
                edgeTable=edgeTable.append(temp)
            #print(nodeFrom)



    edgeTable=edgeTable.drop_duplicates()
    return edgeTable


# %% define wh graph



# %% analyse the traffic of the warehouse
def analyseWhTraffic(D_mov, D_res, G, numPicks=-1, edgePredecessors=True, D_layout=[]):
    #define the traffic chart of the storage system
    #la funzione genera una mappa del traffico (se edgePredecessors=1) e le statistiche di percorrenza delle liste di prelievo



    #get IO nodes
    inputloc=nx.get_node_attributes(G,'input')
    outputloc=nx.get_node_attributes(G,'output')

    #if len(inputloc)>1:
    inputloc=list(inputloc.keys())[0]

    #if len(outputloc)>1:
    outputloc=list(outputloc.keys())[0]

    # check all locations in D_res
    print(f"There are {len(D_mov.loc[~(D_mov.idlocation.isin(D_res.keys()))])} unmapped locations")


    #verifico che le pickinglist siano salvate
    picklists=list(set(D_mov.pickinglist))
    if len(picklists)<2: #setto le pickinglist sugli ordini
        D_mov['pickinglist']=D_mov.ordercode
        picklists=list(set(D_mov.pickinglist))
        print("====WARNING===== No pickinglists recorded. Using ordercode =========")

    if numPicks==-1:
        numPicks=len(picklists)


    cols_res=['pickinglist','distance']
    D_stat_order=pd.DataFrame(columns=cols_res) #statistiche su distanze
    D_arcs=pd.DataFrame(columns=['nodeFrom','nodeTo']) # statistiche su traffico

    #bootstrap pickinglist
    np.random.seed(42)
    pickups = np.random.randint(0,len(picklists) , size=numPicks)
    count=0
    for k in range(0,len(pickups)):
        pick=picklists[pickups[k]]

        count=count+1
        print(f"{count/len(pickups)}")
        D_list=D_mov[D_mov.pickinglist==pick]

        #verifico che tutte le idlocation in pickinglist siano state mappate
        if all(D_list.idlocation.isin(list(D_res.keys()))):
            #scorro tutto la lista degli ordini e costruisco il percorso minimo
            for i in range(0,len(D_list)+1):

                #se e' la prima riga di una lista di prelievo
                if i==0:
                    nodeFrom=inputloc
                    nodeTo=D_res[D_list.idlocation.iloc[i]]
                    if edgePredecessors:
                        path = nx.shortest_path(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    dist = nx.shortest_path_length(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    #print(f"Inizio:{i} From:{nodeFrom},to:{nodeTo}; idlocFrom:IN, idlocto:{D_list.idlocation.iloc[i]}")

                #se e' la ultima riga di una lista di prelievo
                elif i==len(D_list):
                    nodeFrom=D_res[D_list.idlocation.iloc[i-1]]
                    nodeTo=outputloc
                    if edgePredecessors:
                        path = nx.shortest_path(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    dist = nx.shortest_path_length(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    #print(f"Fine:{i} From:{nodeFrom},to:{nodeTo}; idlocFrom:{D_list.idlocation.iloc[i-1]}, idlocto: OUT")
                else:
                    nodeFrom=D_res[D_list.idlocation.iloc[i-1]]
                    nodeTo=D_res[D_list.idlocation.iloc[i]]
                    if edgePredecessors:
                        path = nx.shortest_path(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    dist = nx.shortest_path_length(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    #print(f"transito:{i} From:{nodeFrom},to:{nodeTo}; idlocFrom:{D_list.idlocation.iloc[i-1]}, idlocto: {D_list.idlocation.iloc[i]}")
                #salvo tabella con risultati
                temp=pd.DataFrame([[pick,dist]], columns=cols_res)
                D_stat_order=D_stat_order.append(temp)

                #salvo tabella con archi percorsi
                #print(f"================picklist{count}")
                #print(path)
                if edgePredecessors:
                    for j in range(1,len(path)):
                        temp=pd.DataFrame([[path[j-1], path[j]]],columns=D_arcs.columns)
                        D_arcs=D_arcs.append(temp)

        else:
            print("Idlocations not found:")
            print(pick)
            print(D_list.idlocation.loc[~D_list.idlocation.isin(list(D_res.values()))])

    # group results
    D_stat_picks=D_stat_order.groupby(['pickinglist']).sum()
    mean_dist_picks=np.round(np.mean(D_stat_picks.distance),1)
    std_dist_picks=np.round(np.std(D_stat_picks.distance),1)

    #remove outliers
    #D_stat_picks, perc =cleanUsingIQR(D_stat_picks,['distance'])
    mean_dist_picks=np.round(np.mean(D_stat_picks.distance),1)
    std_dist_picks=np.round(np.std(D_stat_picks.distance),1)

    # draw histogram
    plt.figure()
    plt.hist(D_stat_picks.distance)
    plt.ylabel('N. of picking lists')
    plt.xlabel('Distance')
    plt.title(f"Distance per picking list. {np.round(len(pickups)/len(picklists)*100,1)}% of the dataset ")


    if edgePredecessors:
        #draw traffic chart
        D_stat_arcs=D_arcs.groupby(['nodeFrom','nodeTo']).size().reset_index()
        D_stat_arcs=D_stat_arcs.rename(columns={0:'traffic'})

        #set edge attributes
        edge_attributes_all = {(nodeFrom, nodeTo):{'traffic':0.0001} for (nodeFrom, nodeTo) in G.edges}
        nx.set_edge_attributes(G, edge_attributes_all)

        edge_attributes_traffic = {(nodeFrom, nodeTo):{'traffic':traff} for (nodeFrom, nodeTo, traff) in zip(D_stat_arcs.nodeFrom, D_stat_arcs.nodeTo, D_stat_arcs.traffic)}
        nx.set_edge_attributes(G, edge_attributes_traffic)

        distance=''
        weight='traffic'
        title='Traffic chart'
        arcLabel=False
        nodeLabel=False
        trafficGraph=True
        printNodecoords=True

        dg.printGraph(G, distance, weight, title, arcLabel, nodeLabel, trafficGraph, printNodecoords, D_layout)

    return mean_dist_picks, std_dist_picks, len(D_stat_picks), len(picklists)

# %% Graph pop-dist
def calculateExchangeSaving(D_mov, D_res, G, useSameLevel=False):

    #D_mov e' una tabella di movimenti
    #D_res e' una dizionario di corrispondenze fra idlocation e i nodi del grafo
    # G e' l'oggetto grafo
    #useSameLevel ammette solo scambi fra locazioni dello stesso livello (se True)
    #la funzione calcola i saving determinati dallo scambio fra ubiche

    if useSameLevel:
    #Calcolo la popularity per ogni locazione e livello
        D_bubbles=D_mov.groupby(['idlocation','level']).size().reset_index()

    else :
        D_bubbles=D_mov.groupby(['idlocation']).size().reset_index()
        D_bubbles=pd.DataFrame(D_bubbles)

    D_bubbles=D_bubbles.set_index('idlocation')
    #D_bubbles=pd.DataFrame(D_bubbles)
    D_bubbles=D_bubbles.rename(columns={0:'popularity'})
    D_bubbles['idNode']=None
    D_bubbles['distance']=None
    
    
    
    #calcolo distanze di ogni location
    inputDistance = nx.get_node_attributes(G,'input_distance')
    outputDistance = nx.get_node_attributes(G,'output_distance')
    
    
    for index, row  in D_bubbles.iterrows():
        idNode=D_res[index]
        D_bubbles['idNode'].loc[index]=idNode
        D_bubbles['distance'].loc[index] = inputDistance[idNode] + outputDistance[idNode]
    
    #grafico la distanza di ogni punto dall'IO
    nodecoords = nx.get_node_attributes(G,'coordinates')

    # salvo coordinate
    D_bubbles['loccodex'] = [nodecoords[idNode][0] for idNode in D_bubbles['idNode']]
    D_bubbles['loccodey'] = [nodecoords[idNode][1] for idNode in D_bubbles['idNode']]

    plt.figure()
    plt.scatter(D_bubbles.loccodex, D_bubbles.loccodey, c=D_bubbles.distance)
    plt.colorbar()
    plt.title("Distance of each node from the I/O")

    #D_bubbles, perc =cleanUsingIQR(D_bubbles, ['popularity','distance'])
    #plt.scatter(D_bubbles.distance, D_bubbles.popularity)
    #plt.xlabel("Distance")
    #plt.ylabel("N. of accesses - popularity")

    if useSameLevel:

        #in realta' dovrebbe fare scambi nello stesso livello e anche fra corsie che hanno lo stesso numuero di livelli

        res_cols=['level', 'popularity', 'idNode', 'distance', 'new_idNode', 'new_distance', 'costASIS', 'costTOBE', 'idlocationASIS','idlocationTOBE']
        D_results=pd.DataFrame(columns=res_cols)


        for level in set(D_bubbles.level):
            D_bubbles_level=D_bubbles[D_bubbles.level==level]

            #ordino il dataframe per popularity
            D_bubbles_pop=D_bubbles_level.sort_values(by='popularity', ascending=False)

            #ordino il dataframe per distanza
            D_bubbles_dist=D_bubbles_level.sort_values(by='distance', ascending=True)

            #lavoro sul dataframe di popularity ed identifico in quale loc vorrei prelevare quella popularity
            D_bubbles_pop['new_idNode']=D_bubbles_dist['idNode'].reset_index(drop=True).values
            D_bubbles_pop['new_distance']=D_bubbles_dist['distance'].reset_index(drop=True).values

            #elimino gli zeri da popularity e distanza
            D_bubbles_pop['new_distance']=D_bubbles_pop['new_distance'].replace(0,0.0001)
            D_bubbles_pop['distance']=D_bubbles_pop['distance'].replace(0,0.0001)

            #stimo travel e saving
            D_bubbles_pop['costASIS']=D_bubbles_pop['popularity']*D_bubbles_pop['distance']
            D_bubbles_pop['costTOBE']=D_bubbles_pop['popularity']*D_bubbles_pop['new_distance']

             #salvo scambi di locazioni
            D_bubbles_pop['idlocationASIS']=D_bubbles_pop.index.values
            D_bubbles_pop['idlocationTOBE']=D_bubbles_dist.index.values

            D_results=D_results.append(D_bubbles_pop.reset_index(drop=True))

    else:
        res_cols=['popularity', 'idNode', 'distance', 'new_idNode', 'new_distance', 'costASIS', 'costTOBE']
        D_results=pd.DataFrame(columns=res_cols)

        #ordino il dataframe per popularity
        D_bubbles_pop=D_bubbles.sort_values(by='popularity', ascending=False)

        #ordino il dataframe per distanza
        D_bubbles_dist=D_bubbles.sort_values(by='distance', ascending=True)

        #lavoro sul dataframe di popularity ed identifico in quale loc vorrei prelevare quella popularity
        D_bubbles_pop['new_idNode']=D_bubbles_dist['idNode'].reset_index(drop=True).values
        D_bubbles_pop['new_distance']=D_bubbles_dist['distance'].reset_index(drop=True).values

        #elimino gli zeri da popularity e distanza
        D_bubbles_pop['new_distance']=D_bubbles_pop['new_distance'].replace(0,0.0001)
        D_bubbles_pop['distance']=D_bubbles_pop['distance'].replace(0,0.0001)

        #stimo travel e saving
        D_bubbles_pop['costASIS']=D_bubbles_pop['popularity']*D_bubbles_pop['distance']
        D_bubbles_pop['costTOBE']=D_bubbles_pop['popularity']*D_bubbles_pop['new_distance']

        #salvo scambi di locazioni
        D_bubbles_pop['idlocationASIS']=D_bubbles_pop.index.values
        D_bubbles_pop['idlocationTOBE']=D_bubbles_dist.index.values

        D_results=D_results.append(D_bubbles_pop.reset_index(drop=True))



    D_results=D_results.reset_index(drop=True)


    D_results['saving_rank']=1-D_results['costTOBE']/D_results['costASIS']

    savingTotale=1-sum(D_results['costTOBE'])/sum(D_results['costASIS'])
    #plt.scatter(D_results['new_distance'],D_results['popularity'] )

    D_results['saving_exchange']= D_results['saving_rank']/(sum(D_results['saving_rank']))*savingTotale


    print("=======================================================")
    print(f"The expected saving is: {np.round(savingTotale,3)*100}%")



    #stampa risultati
    D_results['loccodexTOBE'] = [nodecoords[idNode][0] for idNode in D_results['new_idNode']]
    D_results['loccodeyTOBE'] = [nodecoords[idNode][1] for idNode in D_results['new_idNode']]

    D_results.popularity=D_results.popularity.astype(float)#cast popularity
    '''
    #graph as/is
    D_graph=D_results.groupby(['loccodex','loccodey'])['popularity'].agg(['sum']).reset_index()
    plt.figure()
    plt.scatter(D_graph.loccodex, D_graph.loccodey, D_graph['sum'])
    plt.title("Warehouse as-is")
    #graph to/be
    D_graph=D_results.groupby(['loccodexTOBE','loccodeyTOBE'])['popularity'].agg(['sum']).reset_index()
    plt.figure()
    plt.scatter(D_graph.loccodexTOBE, D_graph.loccodeyTOBE, D_graph['sum'])
    plt.title("Warehouse to-be")
    '''
    return D_results


def defineWHgraph(caseStudy, nodecode, idwh, whsubarea, vehicleCategory, draw=False, arcLabel=False, nodeLabel=False, trafficGraph=False):
    # la funzione ritorna un grafo G e una tabella D_res con la corrispondenza ra idlocation e i nodi del grafo

    # importo tutte le coordinate possibili dalla lista movimenti
    
    D_layout=qq.importGraph(caseStudy, nodecode, idwh, whsubarea, vehicleCategory)
    D_layout.columns=D_layout.columns.str.lower()

    D_check=D_layout[['loccodex','loccodey']]

    if len(D_check.drop_duplicates())<2: # se ho una sola coordinata (ad esempio fittizia)
        D_layout=pd.DataFrame() # svuoto il dataframe e riprovo con rack-bay-level


    # stimo eventuali coordinate non  mappate
    if len(D_layout)==0:
        print("===========WARNING: No layout coordinates for this selection")
        print("===========Using an estimation based on rack, bay, level")
        D_layout=qq.importRackBayLevel(caseStudy, nodecode, idwh, whsubarea, vehicleCategory)
        D_layout.columns=D_layout.columns.str.lower()
        D_layout=defineCoordinatesFromRackBayLevel(D_layout)
        D_layout['aislecodex']=None



    allLocs=len(D_layout)

    D_check=D_layout[['loccodex','loccodey']]
    # se ho almeno una coordinata procedo
    if len(D_check.drop_duplicates())>2:

        #importo i punti di IO (scartando i duplicati)
        D_IO=qq.importIO(caseStudy, nodecode, idwh, whsubarea, vehicleCategory).drop_duplicates()
        D_IO.columns=D_IO.columns.str.lower()
        D_IO=D_IO.dropna()
        
        #se il punto di input non e' mappato, viene posizionato al centro
        if len(D_IO[D_IO.inputloc==1]) ==0 :
            idlocation=-1
            loccodey = np.nanmin(D_layout.loccodey) -1
            loccodex = np.nanmean(list(set(D_layout.loccodex)))
            loccodez = 0
            inputloc=1
            outputloc=0
            D_IO=D_IO.append(pd.DataFrame([[idlocation,inputloc,outputloc,loccodex,loccodey,loccodez]],columns=D_IO.columns))
            print(f"=======Input point unmapped. I is set to x:{loccodex},y:{loccodey}")
        
        #se il punto di output non e' mappato, viene posizionato al centro
        if len(D_IO[D_IO.outputloc==1])==0:
            idlocation=-2
            loccodey = np.nanmin(D_layout.loccodey) -1
            loccodex = np.nanmean(list(set(D_layout.loccodex)))
            loccodez = 0
            inputloc=0
            outputloc=1
            D_IO=D_IO.append(pd.DataFrame([[idlocation,inputloc,outputloc,loccodex,loccodey,loccodez]],columns=D_IO.columns))
            print(f"=======Output point unmapped. O is set to x:{loccodex},y:{loccodey}")

        #identifico le fittizie
        D_fake = qq.importFakeLoc(caseStudy, nodecode, idwh, whsubarea, vehicleCategory) # importo fittizie
        D_fake.columns=D_fake.columns.str.lower()
        # identifico il primo punto di IO
        fakecoordx=D_IO.loccodex.iloc[0]
        fakecoordy=D_IO.loccodey.iloc[0]

        # sostituisco le coordinate di tutte le fittizie col punto di IO
        D_layout.loccodex.loc[D_layout.idlocation.isin(D_fake.idlocation)]=fakecoordx
        D_layout.loccodey.loc[D_layout.idlocation.isin(D_fake.idlocation)]=fakecoordy

        # stimo eventuali coordinate delle corsie mancanti
        D_layout=estimateMissingAislecoordX(D_layout)

        #disegna coordinate stimate dopo l'eliminazione dei nulli

        if len(D_layout)==allLocs:
            # trovo corrispondenza fra nodi del grafo e idlocation
            D_nodes, D_res, D_IO = defineGraphNodes(D_layout,D_IO)
    
    
    
            # definisco gli archi
            edgeTable = defineEdgeTable(D_nodes, D_IO)
    
            # define the graph
            G = dg.defineGraph(edgeTable)
    
            # set graph attribute coordinates x and y
            pos = {idlocation:(coordx, coordy) for (idlocation, coordx, coordy) in zip(D_nodes.index.values, D_nodes.aislecodex, D_nodes.loccodey)}
            pos_io = {idlocation:(coordx, coordy) for (idlocation, coordx, coordy) in zip(D_IO.index.values, D_IO.loccodex, D_IO.loccodey)}
            pos.update(pos_io)
            nx.set_node_attributes(G, pos, 'coordinates')
    
            # set boolean input
            attr_io = {idlocation:inputloc  for (idlocation, inputloc) in zip(D_IO.index.values, D_IO.inputloc)}
            nx.set_node_attributes(G, attr_io, 'input')
    
            # set boolean input
            attr_io = {idlocation:outputloc  for (idlocation, outputloc) in zip(D_IO.index.values, D_IO.outputloc)}
            nx.set_node_attributes(G, attr_io, 'output')
            
            # set distance between the nodes and the IO point
            #consider a single input point
            
            idlocation_IN=D_IO[D_IO.inputloc==1].index.values[0]
            
            
            idlocation_OUT=D_IO[D_IO.outputloc==1].index.values[0]
            
           
                
            
            #preparo dataframe con risultati
            D_allNodes=list(G.nodes)
            D_distanceIO=pd.DataFrame(index=D_allNodes)
            D_distanceIO['IN_dist']=None
            D_distanceIO['OUT_dist']=None
            
            #calcolo distanza IO per ogni nodo del grafo
            for index, row in D_distanceIO.iterrows():
                 #distanza IN
                 dist_IN = nx.shortest_path_length(G, source=idlocation_IN, target=index, weight='length', method='dijkstra')
                 dist_OUT = nx.shortest_path_length(G, source=idlocation_OUT, target=index, weight='length', method='dijkstra')
                 D_distanceIO['IN_dist'].loc[index]=dist_IN
                 D_distanceIO['OUT_dist'].loc[index]=dist_OUT
            
            #setto distanza in input
            attr_dist_in = {idlocation:in_dist  for (idlocation, in_dist) in zip(D_distanceIO.index.values, D_distanceIO.IN_dist)}
            nx.set_node_attributes(G, attr_dist_in, 'input_distance')
            
            #setto distanza in output
            attr_dist_out = {idlocation:out_dist  for (idlocation, out_dist) in zip(D_distanceIO.index.values, D_distanceIO.OUT_dist)}
            nx.set_node_attributes(G, attr_dist_out, 'output_distance')
    
    
            #draw graph
    
            if draw:
    
                # print the graph
                distance=weight='length'
                title='Warehouse graph'
                printNodecoords=False
                
                dg.printGraph(G, distance, weight, title, arcLabel, nodeLabel, trafficGraph,printNodecoords , D_layout)
    
            return G, D_res, D_layout
        else:
            print("=======EXIT=======Internal error. Some locations were not mapped")


    else:
        print("======EXIT===== No coordinates mapped to define a graph")