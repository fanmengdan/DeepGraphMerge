import pickle as pkl
import os

def CreateVocab(VertexTypePathList):
    Vocab = set()

    for index in range(0, len(VertexTypePathList)):
        Type_txt = open(VertexTypePathList[index])
        Type_Lines = Type_txt.readlines()

        for line in Type_Lines:
            Vocab.add(line.strip())

    return Vocab

def CreateVocabDict(VertexSet):
    VocabDict = {}
    index = 0

    for item in VertexSet:
        VocabDict[item] = index
        index += 1

    return VocabDict

if __name__ == '__main__':
    reponame = 'glide'
    steps = [2,3,5]
    for step in steps:
        with open(r'./dataset/'+reponame+'/VertexTypePathList/VertexTypePathList_' + str(step) + '.pkl', 'rb') as f:
            VertexTypePathList = pkl.load(f)

        vocabSet = CreateVocab(VertexTypePathList)
        vocabDict = CreateVocabDict(vocabSet)

        filepath = r'./dataset/'+reponame+'/VertexTypeDict/VertexTypeDict_' + str(step) + '.pkl'

        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(filepath, 'wb') as f:
                pkl.dump(vocabDict, f)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")