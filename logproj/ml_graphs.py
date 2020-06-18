# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx



# %% define a graph from an edgetable
def defineGraph(edgeTable):
        edgeFrom = 'nodeFrom'
        edgeTo = 'nodeTo'
        G=nx.from_pandas_edgelist(edgeTable, edgeFrom, edgeTo,edge_attr=True,create_using=nx.Graph())
        return G
    
# %%

def scale_range (series, minimo, massimo):
    series += -(np.min(series))
    series /= np.float(np.max(series)) / (massimo - minimo)
    series += minimo
    return series


def printGraph(G,distance, weight, title, arcLabel=False, nodeLabel=True, trafficGraph=False, printNodecoords=True, D_layout=[]):
    # G is the graph object
        
    # print the coordinates
    fig1 = plt.figure()
    
    if len(D_layout)>0:
        plt.scatter(D_layout.loccodex,D_layout.loccodey,c='black',marker='s',s=1)
        
    #dg.plotGraph(df,edgeFrom,edgeTo,distance,weight,title,arcLabel=False)
    #pos = {idlocation:(coordx, coordy) for (idlocation, coordx, coordy) in zip(D_nodes.index.values, D_nodes.aislecodex, D_nodes.loccodey)}
    #pos_io = {idlocation:(coordx, coordy) for (idlocation, coordx, coordy) in zip(D_IO.index.values, D_IO.loccodex, D_IO.loccodey)}
    #pos.update(pos_io)
    pos= nx.get_node_attributes(G,'coordinates')        
    # represent the graph
    
    if printNodecoords:
        
        
        x=[x for (x,y) in pos.values()]
        y=[y for (x,y) in pos.values()]
        plt.scatter(x,y,c='black',marker='s',s=1)
    
    #edges = G.edges()
    weights = [G[u][v][weight] for u,v in G.edges]
    labels = nx.get_edge_attributes(G,weight)
    
    if pos==[]: #if coordinates are not specified
        pos = nx.layout.spring_layout(G,weight = distance)
    
    
    #node_sizes = weights
    #M = G.number_of_edges()
    
    edge_colors = weights
    edge_width=scale_range(np.float_(weights),1,10)
    
    #edge_alphas = weights
    
    
    plt.title(title)
    nx.draw(G,pos,node_size=0,edge_color='white',with_labels = nodeLabel)
    #nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='Orange')
    if trafficGraph:
        nx.draw_networkx_edges(G, pos, width=edge_width, arrowstyle='->',
                                    edge_color=edge_colors,
                                   edge_cmap=plt.cm.Wistia)
    else:
        nx.draw_networkx_edges(G, pos)
    if arcLabel:
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size =5)
    return fig1
    


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