'''
Prodder loads a trained model's parameters onto a prodder model (that is: an 
abridged version of a full model, which runs only up to the layer which we are 
interested in), as well as the dev data, and generates a pickled list of 
tensors, where the nth tensor represents the output of the prodder model for 
the nth dev example.
'''

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_prodding_args
from collections import OrderedDict
from json import dumps
from models import BiDAFProdder, BiDAF_2Prodder
from os.path import join
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

__MODEL_FROM_NAME = {'baseline-01'      : BiDAFProdder,
                     'baseline-small-01': BiDAFProdder,
                     'exp_02-01'        : BiDAF_2Prodder}

def main(args):

    import gc
    gc.collect()

    _model = __MODEL_FROM_NAME[args.name]
    args.save_dir = f'save/prod/{args.name}_TEST'
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size = 1

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    model = _model(word_vectors=word_vectors,
                   hidden_size=args.hidden_size)
    model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, f'save/train/{args.name}/best.pth.tar', gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    prod_results = []
    log.info(f'Starting prodding on {args.split} split...')

    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            prod_results.append(model(cw_idxs, qw_idxs))
            progress_bar.update(batch_size)

    log.info(f'Results summary: \nlen: {len(prod_results)}\n type: \
               list(tuple({type(prod_results[0][0])}, \
               tuplesize: {len(prod_results[0])}))')

    log.info(f'Clearing memory...')

    del model
    del dataset
    gc.collect()

    log.info(f'Pickling results...')
    with open(f'{args.save_dir}/prod_results.pkl', 'wb') as f:
        pickle.dump(prod_results, f)
    log.info(f'Done pickling.')

if __name__ == '__main__':
    main(get_prodding_args())
