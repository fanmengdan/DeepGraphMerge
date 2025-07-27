import pickle as pkl

def mapHunkID2Num(dicts):
    hunkIDdicts = [] # hunkID: index

    for dict in dicts:
        hunkIDdict = {}

        HunkIDs = dict['allHunkID']
        for index in range(len(HunkIDs)):
            HunkID = HunkIDs[index]
            hunkIDdict[HunkID] = index
        hunkIDdicts.append(hunkIDdict)

    return hunkIDdicts

if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]

    for step in steps:
        with open(r'./dataset/'+reponame+'/HunkIDdict/HunkIDdict_' + str(step) + '.pkl', 'rb') as f:
            HunkID_dict = pkl.load(f)

        hunkIDdicts = mapHunkID2Num(HunkID_dict)
        with open(r'./dataset/'+reponame+'/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'wb') as f:
            pkl.dump(hunkIDdicts, f)
