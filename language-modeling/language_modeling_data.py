"""
This file contains helper functions for loading the patent
data for masked language modeling.
"""

import time
from pathlib import Path
from pprint import pprint
from typing import Optional, Sequence
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _preprocess_dataset_for_language_modeling(dataset: Sequence, sep_token: Optional[int] = None, conditional: bool = True):

    # For filtering out continued patents from our dataset
    decision_to_str = {
        'REJECTED': 0,
        'ACCEPTED': 1,
        'PENDING': 2,
        'CONT-REJECTED': 3,
        'CONT-ACCEPTED': 4,
        'CONT-PENDING': 5
    }

    # Indices of cont patents
    indices_of_cont_patents = {v for k, v in decision_to_str.items() if k.startswith('CONT-')}

    def map_decision_to_string(example):
        return {'decision': decision_to_str[example['decision']]}

    # Performing the remapping means iterating over the dataset
    print('Mapping decision to integer')
    dataset = dataset.map(map_decision_to_string)

    # NOTE: This stores the updated table in a cache file indexed
    # by the current state and the mapping function (I believe)
    print('Processed dataset cached to: ')
    pprint(dataset.cache_files)

    def filter_cont_patents(e):
        return e['decision'] not in indices_of_cont_patents

    def format_example_for_language_modeling(e):

        # Check if claims starts with special text, and if so remove it
        # assert isinstance(e['claims'], list), f'unexpected format for claims: {e["claims"]}'
        if 'What is claimed is:' in e['claims'][:50]:
            e['claims'] = e['claims'].replace('What is claimed is:', '')

        # Format
        # NOTE: The tokenizer will add `bos` and `eos` tokens automatically, so we do not add them here
        if conditional:
            text = 'TITLE {title} {sep} YEAR {year} {sep} IPC {ipc} {sep} CLAIMS {claims}'.format(
                sep=sep_token,
                title=e['title'],
                year=e['filing_date'][:4],
                ipc=e['ipc_label'][:4],
                claims=e['claims'])
        else:
            text = e['claims']
        return {'text': text}

    # Filter out the CONT patents
    print('Filtering out CONT patents')
    print(f'[OLD] len(dataset) = {len(dataset)}')
    dataset = dataset.filter(filter_cont_patents)
    print(f'[NEW] len(dataset) = {len(dataset)}')

    # Format examples
    print('Formatting examples for language modeling')
    dataset = dataset.map(format_example_for_language_modeling, batched=False)
    return dataset


def preprocess_dataset_for_language_modeling(dataset_dict, tokenizer: Optional[PreTrainedTokenizerBase] = None, conditional: bool = False):
    """ Loads dataset for language modeling. Note that the tokenizer is needed in order 
        to add [CLS] and [BOS] tokens to the data. """

    print('****************** Started loading dataset ******************')
    start_time = time.time()

    # Print some metadata
    print('Dataset dictionary contents:')
    pprint(dataset_dict)
    print('Dataset dictionary cached to:')
    pprint(dataset_dict.cache_files)
    print(f'Train dataset initial size: {dataset_dict["train"].shape}')
    print(f'Validation dataset initial size: {dataset_dict["validation"].shape}')

    # Add new tokens to the tokenizer
    print(f'Adding new tokens to tokenizer')
    print(f'[OLD] len(tokenizer.vocab) = {len(tokenizer)}')
    new_tokens = Path('ipc_labels.txt').read_text().splitlines(keepends=False)
    new_tokens += ['TITLE', 'YEAR', 'IPC', 'CLAIMS']
    tokenizer.add_tokens(new_tokens)
    print(f'[NEW] len(tokenizer.vocab) = {len(tokenizer)}')

    # Create training and validation datasets
    print('>>> Training dataset')
    dataset_dict["train"] = _preprocess_dataset_for_language_modeling(
        dataset_dict["train"], 
        sep_token=tokenizer.special_tokens_map['sep_token'],
        conditional=conditional
    )
    print('>>> Validation dataset')
    dataset_dict["validation"] = _preprocess_dataset_for_language_modeling(
        dataset_dict["validation"], 
        sep_token=tokenizer.special_tokens_map['sep_token'],
        conditional=conditional
    )

    print(f'****************** Finished loading dataset in {time.time() - start_time:.1f} seconds ******************')
    return dataset_dict, tokenizer


def _unit_test():
    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    datasets = load_dataset(
        'greeneggsandyaml/test-dataset-debug', "sample",
        train_filing_start_date='2016-01-01',
        train_filing_end_date='2016-01-20',
        val_filing_start_date='2016-01-20',
        val_filing_end_date='2016-01-31',
    )

    # Preprocess dataset
    datasets, tokenizer = preprocess_dataset_for_language_modeling(datasets, tokenizer, conditional=False)
    print(f'Example of claims:')
    print(datasets['train'][42]['claims'])


if __name__ == '__main__':
    _unit_test()