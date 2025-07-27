import pickle
import os
from getPath import get_filelist, filter

if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]

    for step in steps:
        dataset_path = './dataset/'+reponame+'/' + str(step)
        key_word = ['Type']
        # get source and target content
        filelist = get_filelist(dataset_path, [])
        newfilelist = filter(filelist, [], kw = key_word)

        filepath = r'./dataset/'+reponame+'/VertexTypePathList/VertexTypePathList_' + str(step) + '.pkl'

        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(newfilelist, f)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")