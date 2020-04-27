# -*- coding: utf-8 -*-

#https://automating-gis-processes.github.io/CSC/lessons/L1/overview.html

import osmnx as ox
G = ox.graph_from_place('Friuli Venezia giulia, Italy', network_type='drive')
ox.plot_graph(G)
