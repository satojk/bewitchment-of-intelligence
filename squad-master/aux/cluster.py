'''
Cluster
'''

import pickle

import torch
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

def main():

    rle = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 10, 9, 9, 9, 7, 8, 9, 10, 8, 9, 9, 10, 10, 10, 10, 10, 10, 9, 8, 10, 8, 10, 10, 10, 9, 10, 9, 15, 15, 15, 14, 13, 15, 9, 9, 9, 10, 10, 9, 10, 10, 10, 9, 10, 9, 10, 10, 9, 10, 10, 9, 10, 10, 10, 10, 10, 7, 9, 10, 9, 9, 10, 10, 8, 10, 10, 9, 8, 10, 8, 9, 9, 10, 9, 8, 8, 10, 9, 9, 10, 10, 10, 10, 9, 10, 10, 10, 10, 9, 10, 10, 10, 8, 9, 9, 10, 9, 10, 9, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 9, 9, 9, 8, 10, 10, 10, 9, 10, 9, 10, 9, 10, 9, 10, 8, 10, 9, 8, 10, 10, 10, 9, 11, 9, 9, 9, 8, 8, 10, 10, 10, 10, 14, 14, 14, 14, 13, 14, 9, 9, 9, 9, 8, 9, 10, 10, 10, 9, 10, 10, 10, 10, 8, 9, 14, 12, 14, 13, 13, 14, 10, 12, 10, 9, 9, 9, 9, 8, 9, 9, 8, 9, 10, 10, 9, 9, 9, 8, 8, 10, 9, 9, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 10, 9, 9, 9, 9, 15, 14, 15, 14, 15, 15, 15, 13, 15, 14, 15, 14, 14, 13, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 8, 10, 9, 10, 10, 10, 8, 10, 9, 10, 10, 10, 8, 9, 10, 10, 9, 8, 8, 9, 7, 9, 8, 9, 10, 7, 9, 9, 10, 8, 9, 8, 9, 9, 9, 9, 9, 6, 9, 10, 10, 9, 9, 6, 7, 5, 9, 7, 9, 7, 7, 10, 10, 9, 8, 4, 12, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 13, 15, 14, 15, 11, 14, 10, 8, 9, 10, 8, 8, 10, 8, 8, 9, 10, 9, 9, 10, 8, 10, 10, 8, 10, 10, 10, 10, 10, 9, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 10, 10, 10, 9, 9, 9, 9, 9, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 10, 10, 10, 10, 10, 10, 9, 10, 9, 10, 10, 10, 10, 9, 9, 8, 10, 9, 10, 10, 10, 9, 9, 8, 8, 9, 7, 9, 15, 15, 14, 13, 14, 14, 15, 15, 15, 14, 15, 15, 15, 15, 14, 15, 12, 14, 15, 15, 14, 15, 14, 15, 15, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 10, 15, 15, 15, 14, 15, 15, 15, 13, 15, 14, 15, 15, 15, 14, 14, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 9, 20, 20, 20, 18, 20, 19, 15, 15, 14, 15, 10, 10, 9, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    print(f'Unpickling prods...')
    with open(f'prod_results.pkl', 'rb') as f:
        prods = pickle.load(f)
    print(f'Done unpickling.')

    cws = np.empty((0,200)) # np arrays

    print(f'Extracting context words')
    ix = 0
    cur_context_ix = 0
    with tqdm(total=len(prods)) as progress_bar:
        while ix < len(prods):
            context = prods[ix][0]
            cws = np.append(cws, [torch.squeeze(cw).numpy() for cw in torch.split(context, 1, 1)], axis=0)
            ix += rle[cur_context_ix]
            progress_bar.update(rle[cur_context_ix])
            cur_context_ix += 1

    print(ix)
    print(cur_context_ix)
    print(len(rle))
    del prods

    print(f'Running KMeans')
    kmeans = KMeans(n_clusters=100, random_state=0).fit(cws)

    print(f'Pickling...')
    with open('clusters.pkl', 'wb') as f:
        pickle.dump(kmeans.labels_, f)
    print(f'Done pickling...')

if __name__ == '__main__':
    main()
