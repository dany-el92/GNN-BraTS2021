import numpy as np
import os
import networkx as nx
import pickle
import json

'''
A collection of helper functions for graph processing
'''

def save_networkx_graph(G,fp):
    graph_as_json = nx.readwrite.json_graph.node_link_data(G)
    str_dump = json.dumps(graph_as_json)
    with open(fp,'w') as f:
        f.write(str_dump)
    #print("Saved ",fp)
