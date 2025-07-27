import pickle as pkl

def mapGroupHunkID2Num(dicts):

    for dict in dicts:
        HunkIDs = dict['allHunkID']

        for key in dict:
            if "group" in key:
                for index in range(len(dict[key])):
                    hunkid = dict[key][index]
                    dict[key][index] = HunkIDs.index(hunkid)

        for index in range(len(HunkIDs)):
            HunkIDs[index] = index

    return dicts

if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]

    for step in steps:
        with open(r'./dataset/'+reponame+'/HunkIDdict/HunkIDdict_' + str(step) + '.pkl', 'rb') as f:
            HunkID_dict = pkl.load(f)

        newDict = mapGroupHunkID2Num(HunkID_dict)
        with open(r'./dataset/'+reponame+'/HunkIDdict/NewHunkIDdict_' + str(step) + '.pkl', 'wb') as f:
            pkl.dump(newDict, f)