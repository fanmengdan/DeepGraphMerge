import pickle
import networkx as nx
import numpy as np
import os

def build_graph(vertexlist, sourceList, targetList):
    Graph = nx.DiGraph()

    for index in range(len(vertexlist)):
        Graph.add_node(vertexlist[index]-1)

    for index in range(len(sourceList)):
        Graph.add_edge(sourceList[index]-1, targetList[index]-1)

    return Graph

def build_Adjs(VertexPathList,VertexTypePathList,VertexTypeDict):
    Adjs = []

    for index in range(0, len(VertexPathList)-2, 3):

        Id_txt = open(VertexPathList[index])
        Source_txt = open(VertexPathList[index+1])
        Target_txt = open(VertexPathList[index+2])
        Type_txt = open(VertexTypePathList[index//3])

        IdLines = Id_txt.readlines()
        SourceLines = Source_txt.readlines()
        TargetLines = Target_txt.readlines()
        TypeLines = Type_txt.readlines()

        idList = []
        soureList = []
        targetList = []
        typeList = []

        for index in range(len(IdLines)):
            idList.append(float(IdLines[index].strip()))
            typeList.append(TypeLines[index].strip())

        for index in range(len(SourceLines)):
            soureList.append(float(SourceLines[index].strip()))
            targetList.append(float(TargetLines[index].strip()))

        if len(idList) !=0:
            graph = build_graph(idList, soureList, targetList)

            Adj = np.array(nx.adjacency_matrix(graph).todense(), dtype=float)

            row, col = np.diag_indices_from(Adj)

            nodeAttributeList = []
            rowlist = list(row)
            for item in rowlist:
                key = typeList[item]
                value = VertexTypeDict[key]
                nodeAttributeList.append(value)

            Adj[row,col] = nodeAttributeList

            Adjs.append(Adj)
        else:
            print('Empty commit:', VertexPathList[index])

    return Adjs


if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]

    for step in steps:
        with open(r'./dataset/'+reponame+'/VertexPathList/VertexPathList_' + str(step) + '.pkl', 'rb') as f:
            VertexPathList = pickle.load(f)

        with open(r'./dataset/'+reponame+'/VertexTypePathList/VertexTypePathList_' + str(step) + '.pkl', 'rb') as f:
            VertexTypePathList = pickle.load(f)

        with open(r'./dataset/'+reponame+'/VertexTypeDict/VertexTypeDict_' + str(step) + '.pkl', 'rb') as f:
            VertexTypeDict = pickle.load(f)

        Adjs = build_Adjs(VertexPathList, VertexTypePathList, VertexTypeDict)

        filepath = r'./Adjset/'+reponame+'/Adj/Adjs_' + str(step) + '.npy'

        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(filepath, 'wb') as f:
                np.save(filepath, Adjs)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")