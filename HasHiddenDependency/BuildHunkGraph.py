import pickle as pkl
import networkx as nx
import numpy as np
import itertools
import os


def build_Adjs(dicts):
    Adjs = []

    for dict in dicts:
        Graph = nx.Graph()

        HunkIDs = dict['allHunkID']
        for item in HunkIDs:
            Graph.add_node(item)

        for key in dict:
            if "group" in key:
                sourceList = dict[key]
                edges = list(itertools.product(sourceList, sourceList))
                for item in edges:
                    Graph.add_edge(item[0],item[1])

        adj = np.array(nx.adjacency_matrix(Graph).todense(), dtype=float)
        Adj = adj - np.eye(adj.shape[0])

        Adjs.append(Adj)

    return Adjs


if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]
    for step in steps:
        with open(r'./dataset/'+ reponame +'/HunkIDdict/NewHunkIDdict_' + str(step) + '.pkl', 'rb') as f:
            HunkIDdict = pkl.load(f)

        HunkAdjs = build_Adjs(HunkIDdict)

        filepath = r'./Adjset/'+reponame+'/Adj/HunkAdjs_' + str(step) + '.npy'

        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(filepath, 'wb') as f:
                np.save(filepath, HunkAdjs)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")