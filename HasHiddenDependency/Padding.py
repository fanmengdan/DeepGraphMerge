import numpy as np
import os

if __name__ == '__main__':
    reponame = 'glide'
    steps = [2, 3, 5]

    for step in steps:
        Adjs = np.load('./Adjset/'+ reponame +'/Adj/Adjs_' + str(step) + '.npy', allow_pickle=True)

        rows = []
        for array in Adjs:
            rows.append(array.shape[0])
        max_row = max(rows)

        PAdjs = []
        for array in Adjs:
            diff = max_row-array.shape[0]
            PAdj = np.pad(array,
                           ((0, diff),
                            (0, diff)),
                           'constant')
            PAdjs.append(PAdj)

        HunkAdjs = np.load('./Adjset/'+ reponame +'/Adj/HunkAdjs_' + str(step) + '.npy', allow_pickle=True)

        rows1 = []
        for array in HunkAdjs:
            rows1.append(array.shape[0])
        max_row1 = max(rows1)

        PHunkAdjs = []
        for array in HunkAdjs:
            diff1 = max_row1-array.shape[0]
            PHunkAdj = np.pad(array,
                           ((0, diff1),
                            (0, diff1)),
                           'constant')
            PHunkAdjs.append(PHunkAdjs)

        PAdjs_path = './Adjset/' + reponame + '/Padding_Adjs/PAdjs_' + str(step) + '.npy'
        PHunkAdjs_path = './Adjset/'+ reponame +'/Padding_Adjs/PHunkAdjs_' + str(step) + '.npy'

        directory = os.path.dirname(PAdjs_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory2 = os.path.dirname(PHunkAdjs_path)
        if not os.path.exists(directory2):
            os.makedirs(directory2)

        try:
            with open(PAdjs_path, 'wb') as f:
                np.save(PAdjs_path, PAdjs)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            with open(PHunkAdjs_path, 'wb') as f:
                np.save(PHunkAdjs_path, PHunkAdjs)
            print("File saved successfully.")
        except FileNotFoundError:
            print("Directory creation failed. Please check permissions and path.")
        except Exception as e:
            print(f"An error occurred: {e}")