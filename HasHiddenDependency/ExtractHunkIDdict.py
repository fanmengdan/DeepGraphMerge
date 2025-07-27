import pickle as pkl
import os

def readHunkID(path):
    with open(path, 'rb') as f:
        GTPathList = pkl.load(f)

    AllGT = []

    for index1 in range(0, len(GTPathList)):
        namePathParts = GTPathList[index1].split('//')
        name = namePathParts[-1]

        GT_txt = open(GTPathList[index1])
        GT_Lines = GT_txt.readlines()
        HunkIDs_AllGroup = []
        HunkIDs_Dict = {}
        i = 1

        for line in GT_Lines:
            line = line.strip()

            parts = line.split(":")

            HunkIDs_oneGroup = parts[-1].split(',')

            for index3 in range(len(HunkIDs_oneGroup)):
                HunkIDs_oneGroup[index3] = HunkIDs_oneGroup[index3].strip()

            HunkIDs_oneGroup[0] = HunkIDs_oneGroup[0].lstrip('[')
            HunkIDs_oneGroup[-1] = HunkIDs_oneGroup[-1].rstrip(']')

            HunkIDs_AllGroup.extend(HunkIDs_oneGroup)

            HunkIDs_Dict['group_'+ str(i)] = HunkIDs_oneGroup
            i += 1

        HunkIDs_Dict['name'] = name
        HunkIDs_Dict['allHunkID'] = HunkIDs_AllGroup
        AllGT.append(HunkIDs_Dict)

    return AllGT

if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]
    for step in steps:
        path = r'./dataset/'+reponame+'/GTPathList/GTPathList_' + str(step) + '.pkl'

        HunkIDs_dict = readHunkID(path)

        filepath = r'./dataset/'+reponame+'/HunkIDdict/HunkIDdict_' + str(step) + '.pkl'

        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(filepath, 'wb') as f:
                pkl.dump(HunkIDs_dict, f)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")