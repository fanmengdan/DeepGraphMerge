import os

def get_filelist(dir, Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)

    elif os.path.isdir(dir):
        for subDir in os.listdir(dir):
            newDir = os.path.join(dir, subDir)
            get_filelist(newDir, Filelist)

    return Filelist

def filter (Filelist, newfilelist, kw):
    for item in Filelist:
        if any(word if word in item else False for word in kw):
            newfilelist.append(item)
    return newfilelist