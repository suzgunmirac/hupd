![HUPD-Diagram](https://github.com/suzgunmirac/hupd/blob/main/figures/HUPD-Logo.png)

# The Harvard USPTO Patent Dataset (HUPD)

## Multi-Class IPC/CPC Classification (Automated Subject Classification)

This directory contains code, data splits, and general information about the task of multi-class IPC/CPC classification (also known as, automated subject classification or patent classification. We support both training from scratch and finetuning pretranied models.

Some of the key data arguments include:
* `--section (default: 'abstract')`: "Patent application section of interest." 
* `--train_filing_start_date (default: '2011-01-01')`: "Start date for filtering patents training data."
* `--val_filing_end_date (default: '2016-12-31')`: "End date for filtering patents validation data."
* `--uniform split (default: false)`: "Uniformly split the data into training and validation sets."


For example, here is how you can train a RoBERTa model (`roberta-base`):
```bash
mkdir -p models
mkdir -p results
python main.py --batch_size 64 \
--section abstract \
--train_filing_start_date 2011-01-01 \
--val_filing_end_date 2016-12-31 \
--max_length 256 \
--epoch_n 5 \
--val_every 5000 \
--model_name roberta-base \
--lr 2e-5 \
--uniform_split \
--save_path models/multilabel_ipc_roberta_abstract_train2011to16 \
--tokenizer_save_path models/multilabel_ipc_roberta_abstract_train2011to16_tokenizer \
--filename results/multilabel_ipc_roberta_abstract_maxlength256.txt \
--wandb
```

Here is an example of how you can do inference using a `distilroberta-base` model:
```bash
python main.py \
    --batch_size 64 \
    --section abstract \
    --train_filing_start_date 2011-01-01 \
    --val_filing_end_date 2016-12-31 \
    --uniform_split \
    --min_frequency 3 \
    --max_length 256 \
    --epoch_n -1 \
    --model_name roberta-base \
    --model_path models/multilabel_ipc_roberta_abstract_train2011to16 \
    --tokenizer_path models/multilabel_ipc_roberta_abstract_train2011to16_tokenizer \
    --filename test.txt \
    --validation --val_set_balancer
```

Finally, here is the full help output for `main.py`:
```
usage: main.py [-h] [--cache_dir CACHE_DIR] [--data_dir DATA_DIR]
               [--dataset_load_path DATASET_LOAD_PATH] [--section SECTION]
               [--train_filing_start_date TRAIN_FILING_START_DATE]
               [--train_filing_end_date TRAIN_FILING_END_DATE]
               [--val_filing_start_date VAL_FILING_START_DATE]
               [--val_filing_end_date VAL_FILING_END_DATE]
               [--vocab_size VOCAB_SIZE] [--min_frequency MIN_FREQUENCY]
               [--max_length MAX_LENGTH] [--use_wsampler] [--val_set_balancer]
               [--uniform_split] [--train_from_scratch] [--validation]
               [--batch_size BATCH_SIZE] [--epoch_n EPOCH_N]
               [--val_every VAL_EVERY] [--lr LR] [--eps EPS] [--wandb]
               [--wandb_name WANDB_NAME] [--pos_class_weight POS_CLASS_WEIGHT]
               [--use_scheduler] [--filename FILENAME]
               [--np_filename NP_FILENAME] [--model_name MODEL_NAME]
               [--embed_dim EMBED_DIM] [--tokenizer_path TOKENIZER_PATH]
               [--model_path MODEL_PATH] [--save_path SAVE_PATH]
               [--tokenizer_save_path TOKENIZER_SAVE_PATH]
               [--n_filters N_FILTERS]
               [--filter_sizes FILTER_SIZES [FILTER_SIZES ...]]
               [--dropout DROPOUT] [--naive_bayes_version NAIVE_BAYES_VERSION]
               [--alpha_smooth_val ALPHA_SMOOTH_VAL] [--topk TOPK]

optional arguments:
  -h, --help            show this help message and exit
  --cache_dir CACHE_DIR
                        Cache directory.
  --data_dir DATA_DIR   Patent data directory.
  --dataset_load_path DATASET_LOAD_PATH
                        Patent data main data load path (viz., ../patents.py).
  --section SECTION     Patent application section of interest.
  --train_filing_start_date TRAIN_FILING_START_DATE
                        Start date for filtering the training data.
  --train_filing_end_date TRAIN_FILING_END_DATE
                        End date for filtering the training data.
  --val_filing_start_date VAL_FILING_START_DATE
                        Start date for filtering the training data.
  --val_filing_end_date VAL_FILING_END_DATE
                        End date for filtering the validation data.
  --vocab_size VOCAB_SIZE
                        Vocabulary size (of the tokenizer).
  --min_frequency MIN_FREQUENCY
                        The minimum frequency that a token/word needs to have
                        in order to appear in the vocabulary.
  --max_length MAX_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this number will
                        be trunacated.
  --use_wsampler        Use a weighted sampler (for the training set).
  --val_set_balancer    Use a balanced set for validation? That is, do you
                        want the same number of classes of examples in the
                        validation set.
  --uniform_split       Uniformly split the data into training and validation
                        sets.
  --train_from_scratch  Train the model from the scratch.
  --validation          Perform only validation/inference. (No performance
                        evaluation on the training data necessary).
  --batch_size BATCH_SIZE
                        Batch size.
  --epoch_n EPOCH_N     Number of epochs (for training).
  --val_every VAL_EVERY
                        Number of iterations we should take to perform
                        validation.
  --lr LR               Model learning rate.
  --eps EPS             Epsilon value for the learning rate.
  --wandb               Use wandb.
  --wandb_name WANDB_NAME
                        wandb project name.
  --pos_class_weight POS_CLASS_WEIGHT
                        The class weight of the rejected class label (it is 0
                        by default).
  --use_scheduler       Use a scheduler.
  --filename FILENAME   Name of the results file to be saved.
  --np_filename NP_FILENAME
                        Name of the numpy file to be saved.
  --model_name MODEL_NAME
                        Name of the model.
  --embed_dim EMBED_DIM
                        Embedding dimension of the model.
  --tokenizer_path TOKENIZER_PATH
                        (Pre-trained) tokenizer path.
  --model_path MODEL_PATH
                        (Pre-trained) model path.
  --save_path SAVE_PATH
                        The path where the model is going to be saved.
  --tokenizer_save_path TOKENIZER_SAVE_PATH
                        The path where the tokenizer is going to be saved.
  --n_filters N_FILTERS
                        Number of filters in the CNN (if applicable)
  --filter_sizes FILTER_SIZES [FILTER_SIZES ...]
                        Filter sizes for the CNN (if applicable).
  --dropout DROPOUT     Use dropout for the CNN model (if applicable)
  --naive_bayes_version NAIVE_BAYES_VERSION
                        Type of the Naive Bayes classifer (if applicable).
  --alpha_smooth_val ALPHA_SMOOTH_VAL
                        Alpha smoothing value for the Naive Bayes classifier
                        (if applicable).
  --topk TOPK           TOPK accuracy
```