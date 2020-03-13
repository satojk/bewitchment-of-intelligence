'''
Prodder loads a trained model's parameters onto a prodder model (that is: an 
abridged version of a full model, which runs only up to the layer which we are 
interested in), as well as the dev data, and generates a pickled list of 
tensors, where the nth tensor represents the output of the prodder model for 
the nth dev example.
'''

import pickle
import spacy

import util

from args import get_prodding_args
from collections import OrderedDict
from json import dumps
from os.path import join
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

def main(args):

    args.save_dir = f'save/tag/{args.name}_TEST'
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Get model
    nlp = spacy.load('en_core_web_sm')

    # Get data
    log.info('Building dataset...')
    with open('data/test_eval.json', 'r') as f:
        dataset = json_load(f)

    tag_results = []
    log.info(f'Starting tagging on {args.split} split...')

    with tqdm(total=len(dataset)) as progress_bar:
        for i in range(len(dataset)):
            k = str(i+1)
            context = nlp(dataset[k]['context'])
            question = nlp(dataset[k]['question'])
            tag_results.append((context, question))
            progress_bar.update(1)

    log.info(f'Results summary: \nlen: {len(tag_results)}\n type: \
               list(tuple({type(tag_results[0][0])}, \
               tuplesize: {len(tag_results[0])}))')

    log.info(f'Pickling results...')
    with open(f'{args.save_dir}/tag_results.pkl', 'wb') as f:
        pickle.dump(tag_results, f)
    log.info(f'Done pickling.')

if __name__ == '__main__':
    main(get_prodding_args())
