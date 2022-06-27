"""
Here is an example of how one might load this dataset, explore it, 
transform it, and finally load it in PyTorch.
"""

import torch
from pprint import pprint
from datasets import load_dataset

# ----- Data Loading ------
# 
# Note that if you have already downloaded the data into your preferred
# directory, you can specify its location with the following arguments:
#   data_files="/path/to/metadata-2021-02-10.feather"
#   data_dir="path/to/json-files"
# 
# The following line will download a small version of the dataset for
# you. If you want to download the entire dataset, use name="all"
# instead of name="sample".
dataset_dict = load_dataset(
    'greeneggsandyaml/test-dataset-debug', 
    name="sample",
    ipcr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-20',
    val_filing_start_date='2016-01-20',
    val_filing_end_date='2016-01-31',
)

# Here we can see the `train` and `val` splits, along with the
# location of the cached data files
print('Dataset contents:')
print(dataset_dict)

print('Dataset cache location:')
print(dataset_dict.cache_files)

# Data
train_dataset = dataset_dict["train"]
val_dataset = dataset_dict["validation"]
print(f'Train dataset shape: {train_dataset.shape}')
print(f'Validation dataset shape: {val_dataset.shape}')

# List all available fields
print(f'Dataset fields:')
print(train_dataset.column_names)

# Example: preprocess the abstract field of the dataset
# using HF tokenizers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# We tokenize in batches, so tokenization is quite fast
train_dataset = train_dataset.map(
    lambda e: tokenizer(e['abstract'], truncation=True, padding='max_length'),
    batched=True,
    desc="Tokenizing training files"
)
val_dataset = val_dataset.map(
    lambda e: tokenizer(e['abstract'], truncation=True, padding='max_length'),
    batched=True,
    desc="Tokenizing training files"
)

# Since we've tokenized the dataset, we have a new cache location
print('Dataset cache location after tokenization:')
print(train_dataset.cache_files)

# And we have added some fields to our dataset
print('Dataset fields after tokenization:')
print(train_dataset.column_names)

# Convert to PyTorch Dataset
# NOTE: If you also want to return string columns (as a list), just
# pass `output_all_columns=True` to the dataset
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Standard PyTorch DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2)
print('Shapes of items in batch:')
pprint({k: v.shape for k, v in next(iter(train_dataloader)).items()})
print('Example of first 10 input_ids from batch:')
batch = next(iter(train_dataloader))
pprint(batch['input_ids'][0, :10])
print('Decoded input_ids from batch:')
print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))

print('All done. Enjoy the dataset!')