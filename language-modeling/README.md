![HUPD-Diagram](https://github.com/suzgunmirac/hupd/blob/main/figures/HUPD-Logo.png)

# The Harvard USPTO Patent Dataset (HUPD)

## Language Modeling

This directory contains code, data splits, and general information about the language modeling task. The code is adapted from the standard [Huggingface language modeling example](https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/language-modeling/run_mlm.py). We support both training from scratch and finetuning pretranied models for masked language modeling (BERT, ALBERT, RoBERTa, etc.). 

You can use `--conditional` to specify whether you want to train a conditional model, in which the model is conditioned on the patent filing metadata (i.e. the IPC class, title, etc.). The default is an unconditional model.

Some of the key data arguments include:
* `--ipcr_label (default: None)`: "IPCR label for filtering patents data"
* `--train_filing_start_date (default: '2011-01-01')`: "Start date for filtering patents training data"
* `--train_filing_end_date (default: '2016-12-31')`: "End date for filtering patents training data"
* `--val_filing_start_date (default: '2016-12-31')`: "Start date for filtering patents validation data"
* `--val_filing_end_date (default: '2017-12-31')`: "End date for filtering patents validation data"
* `--use_sample_data (default: True)`: "Use a small sample of the patent data (useful for debugging)"

Our standard split is the following:
```bash
  --train_filing_start_date "2011-01-01"
  --train_filing_end_date "2017-01-01"
  --val_filing_start_date "2017-01-01"
  --val_filing_end_date "2018-01-01"
```

For example, here is how you can train a `distilroberta-base` model:
```bash
mkdir -p outputs/patents-distilroberta-base-unconditional
python main.py \
    --run_name my-run \
    --model_name_or_path distilroberta-base \
    --dataset_name patents \
    --do_train \
    --do_eval \
    --warmup_steps=10000 \
    --num_train_epochs 5 \
    --save_total_limit 5 \
    --save_steps 5000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --output_dir outputs/patents-distilroberta-base-unconditional \
    --train_filing_start_date "2011-01-01" \
    --train_filing_end_date "2017-01-01" \
    --val_filing_start_date "2017-01-01" \
    --val_filing_end_date "2018-01-01" 
```

Finally, here is the full help output for `main.py`:
```
usage: main.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
               [--model_type MODEL_TYPE] [--config_name CONFIG_NAME]
               [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]
               [--no_use_fast_tokenizer]
               [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
               [--model_revision MODEL_REVISION]
               [--use_auth_token [USE_AUTH_TOKEN]]
               [--dataset_name DATASET_NAME]
               [--dataset_config_name DATASET_CONFIG_NAME]
               [--overwrite_cache [OVERWRITE_CACHE]]
               [--validation_split_percentage VALIDATION_SPLIT_PERCENTAGE]
               [--max_seq_length MAX_SEQ_LENGTH]
               [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
               [--mlm_probability MLM_PROBABILITY]
               [--line_by_line [LINE_BY_LINE]]
               [--pad_to_max_length [PAD_TO_MAX_LENGTH]] [--data_dir DATA_DIR]
               [--data_files DATA_FILES] [--ipcr_label IPCR_LABEL]
               [--train_filing_start_date TRAIN_FILING_START_DATE]
               [--train_filing_end_date TRAIN_FILING_END_DATE]
               [--val_filing_start_date VAL_FILING_START_DATE]
               [--val_filing_end_date VAL_FILING_END_DATE]
               [--conditional [CONDITIONAL]] [--no_use_sample_data]
               [--use_sample_data [USE_SAMPLE_DATA]] --output_dir OUTPUT_DIR
               [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
               [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
               [--do_predict [DO_PREDICT]]
               [--evaluation_strategy {no,steps,epoch}]
               [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
               [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
               [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
               [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
               [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
               [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
               [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
               [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
               [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
               [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
               [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS]
               [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
               [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS]
               [--log_level {debug,info,warning,error,critical,passive}]
               [--log_level_replica {debug,info,warning,error,critical,passive}]
               [--no_log_on_each_node] [--log_on_each_node [LOG_ON_EACH_NODE]]
               [--logging_dir LOGGING_DIR]
               [--logging_strategy {no,steps,epoch}]
               [--logging_first_step [LOGGING_FIRST_STEP]]
               [--logging_steps LOGGING_STEPS]
               [--save_strategy {no,steps,epoch}] [--save_steps SAVE_STEPS]
               [--save_total_limit SAVE_TOTAL_LIMIT]
               [--save_on_each_node [SAVE_ON_EACH_NODE]] [--no_cuda [NO_CUDA]]
               [--seed SEED] [--fp16 [FP16]] [--fp16_opt_level FP16_OPT_LEVEL]
               [--fp16_backend {auto,amp,apex}]
               [--fp16_full_eval [FP16_FULL_EVAL]] [--local_rank LOCAL_RANK]
               [--tpu_num_cores TPU_NUM_CORES]
               [--tpu_metrics_debug [TPU_METRICS_DEBUG]] [--debug DEBUG]
               [--dataloader_drop_last [DATALOADER_DROP_LAST]]
               [--eval_steps EVAL_STEPS]
               [--dataloader_num_workers DATALOADER_NUM_WORKERS]
               [--past_index PAST_INDEX] [--run_name RUN_NAME]
               [--disable_tqdm DISABLE_TQDM] [--no_remove_unused_columns]
               [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
               [--label_names LABEL_NAMES [LABEL_NAMES ...]]
               [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
               [--metric_for_best_model METRIC_FOR_BEST_MODEL]
               [--greater_is_better GREATER_IS_BETTER]
               [--ignore_data_skip [IGNORE_DATA_SKIP]]
               [--sharded_ddp SHARDED_DDP] [--deepspeed DEEPSPEED]
               [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
               [--adafactor [ADAFACTOR]] [--group_by_length [GROUP_BY_LENGTH]]
               [--length_column_name LENGTH_COLUMN_NAME]
               [--report_to REPORT_TO [REPORT_TO ...]]
               [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
               [--no_dataloader_pin_memory]
               [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
               [--no_skip_memory_metrics]
               [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
               [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
               [--push_to_hub [PUSH_TO_HUB]]
               [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
               [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID]
               [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
               [--push_to_hub_token PUSH_TO_HUB_TOKEN]
               [--mp_parameters MP_PARAMETERS]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization.Don't
                        set if you want to train a model from scratch.
                        (default: None)
  --model_type MODEL_TYPE
                        If training from scratch, pass a model type from the
                        list: rembert, roformer, big_bird, wav2vec2, convbert,
                        ibert, mobilebert, distilbert, albert, camembert, xlm-
                        roberta, mbart, megatron-bert, mpnet, bart, reformer,
                        longformer, roberta, deberta-v2, deberta, flaubert,
                        squeezebert, bert, xlm, electra, funnel, layoutlm,
                        tapas (default: None)
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name (default: None)
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name (default: None)
  --cache_dir CACHE_DIR
                        Where do you want to store the pretrained models
                        downloaded from huggingface.co (default: None)
  --no_use_fast_tokenizer
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not. (default: True)
  --use_fast_tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not. (default: True)
  --model_revision MODEL_REVISION
                        The specific model version to use (can be a branch
                        name, tag name or commit id). (default: main)
  --use_auth_token [USE_AUTH_TOKEN]
                        Will use the token generated when running
                        `transformers-cli login` (necessary to use this script
                        with private models). (default: False)
  --dataset_name DATASET_NAME
                        The name of the dataset to use.. (default: patents)
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the
                        datasets library). (default: None)
  --overwrite_cache [OVERWRITE_CACHE]
                        Overwrite the cached training and evaluation sets
                        (default: False)
  --validation_split_percentage VALIDATION_SPLIT_PERCENTAGE
                        The percentage of the train set used as validation set
                        in case there's no validation split (default: 5)
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated. (default: None)
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing.
                        (default: None)
  --mlm_probability MLM_PROBABILITY
                        Ratio of tokens to mask for masked language modeling
                        loss (default: 0.15)
  --line_by_line [LINE_BY_LINE]
                        Whether distinct lines of text in the dataset are to
                        be handled as distinct sequences. (default: False)
  --pad_to_max_length [PAD_TO_MAX_LENGTH]
                        Whether to pad all samples to `max_seq_length`. If
                        False, will pad the samples dynamically when batching
                        to the maximum length in the batch. (default: False)
  --data_dir DATA_DIR   Patent data directory (default: None)
  --data_files DATA_FILES
                        Patent data directory (default: None)
  --ipcr_label IPCR_LABEL
                        IPCR label for filtering patents data (default: None)
  --train_filing_start_date TRAIN_FILING_START_DATE
                        Start date for filtering patents training data
                        (default: 2011-01-01)
  --train_filing_end_date TRAIN_FILING_END_DATE
                        End date for filtering patents training data (default:
                        2017-01-01)
  --val_filing_start_date VAL_FILING_START_DATE
                        Start date for filtering patents validation data
                        (default: 2017-01-01)
  --val_filing_end_date VAL_FILING_END_DATE
                        End date for filtering patents validation data
                        (default: 2018-01-01)
  --conditional [CONDITIONAL]
                        Metadata-conditional language modeling (default:
                        False)
  --no_use_sample_data  Use sample of patent data (useful for debugging)
                        (default: True)
  --use_sample_data [USE_SAMPLE_DATA]
                        Use sample of patent data (useful for debugging)
                        (default: True)
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory.Use this
                        to continue training if output_dir points to a
                        checkpoint directory. (default: False)
  --do_train [DO_TRAIN]
                        Whether to run training. (default: False)
  --do_eval [DO_EVAL]   Whether to run eval on the dev set. (default: False)
  --do_predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default:
                        False)
  --evaluation_strategy {no,steps,epoch}
                        The evaluation strategy to use. (default: no)
  --prediction_loss_only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only
                        returns the loss. (default: False)
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for training.
                        (default: 8)
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for evaluation.
                        (default: 8)
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training. (default: None)
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred.Batch size per GPU/TPU core/CPU for
                        evaluation. (default: None)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass. (default: 1)
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before
                        moving the tensors to the CPU. (default: None)
  --learning_rate LEARNING_RATE
                        The initial learning rate for AdamW. (default: 5e-05)
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some. (default:
                        0.0)
  --adam_beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer (default: 0.9)
  --adam_beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer (default: 0.999)
  --adam_epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer. (default: 1e-08)
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm. (default: 1.0)
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default:
                        3.0)
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs. (default: -1)
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use. (default: linear)
  --warmup_ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total
                        steps. (default: 0.0)
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps. (default: 0)
  --log_level {debug,info,warning,error,critical,passive}
                        Logger log level to use on the main node. Possible
                        choices are the log levels as strings: 'debug',
                        'info', 'warning', 'error' and 'critical', plus a
                        'passive' level which doesn't set anything and lets
                        the application set the level. Defaults to 'passive'.
                        (default: passive)
  --log_level_replica {debug,info,warning,error,critical,passive}
                        Logger log level to use on replica nodes. Same choices
                        and defaults as ``log_level`` (default: passive)
  --no_log_on_each_node
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: True)
  --log_on_each_node [LOG_ON_EACH_NODE]
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: True)
  --logging_dir LOGGING_DIR
                        Tensorboard log dir. (default: None)
  --logging_strategy {no,steps,epoch}
                        The logging strategy to use. (default: steps)
  --logging_first_step [LOGGING_FIRST_STEP]
                        Log the first global_step (default: False)
  --logging_steps LOGGING_STEPS
                        Log every X updates steps. (default: 500)
  --save_strategy {no,steps,epoch}
                        The checkpoint save strategy to use. (default: steps)
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps. (default: 500)
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints.Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints (default: None)
  --save_on_each_node [SAVE_ON_EACH_NODE]
                        When doing multi-node distributed training, whether to
                        save models and checkpoints on each node, or only on
                        the main one (default: False)
  --no_cuda [NO_CUDA]   Do not use CUDA even when it is available (default:
                        False)
  --seed SEED           Random seed that will be set at the beginning of
                        training. (default: 42)
  --fp16 [FP16]         Whether to use 16-bit (mixed) precision instead of
                        32-bit (default: False)
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html (default: O1)
  --fp16_backend {auto,amp,apex}
                        The backend to be used for mixed precision. (default:
                        auto)
  --fp16_full_eval [FP16_FULL_EVAL]
                        Whether to use full 16-bit precision evaluation
                        instead of 32-bit (default: False)
  --local_rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script) (default: None)
  --tpu_metrics_debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is
                        preferred. TPU: Whether to print debug metrics
                        (default: False)
  --debug DEBUG         Whether or not to enable debug mode. Current options:
                        `underflow_overflow` (Detect underflow and overflow in
                        activations and weights), `tpu_metrics_debug` (print
                        debug metrics on TPU). (default: )
  --dataloader_drop_last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible
                        by the batch size. (default: False)
  --eval_steps EVAL_STEPS
                        Run an evaluation every X steps. (default: None)
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading
                        (PyTorch only). 0 means that the data will be loaded
                        in the main process. (default: 0)
  --past_index PAST_INDEX
                        If >=0, uses the corresponding part of the output as
                        the past state for next step. (default: -1)
  --run_name RUN_NAME   An optional descriptor for the run. Notably used for
                        wandb logging. (default: None)
  --disable_tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars.
                        (default: None)
  --no_remove_unused_columns
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: True)
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: True)
  --label_names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels. (default: None)
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during
                        training at the end of training. (default: False)
  --metric_for_best_model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
                        (default: None)
  --greater_is_better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not. (default: None)
  --ignore_data_skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data. (default: False)
  --sharded_ddp SHARDED_DDP
                        Whether or not to use sharded DDP training (in
                        distributed training only). The base option should be
                        `simple`, `zero_dp_2` or `zero_dp_3` and you can add
                        CPU-offload to `zero_dp_2` or `zero_dp_3` like this:
                        zero_dp_2 offload` or `zero_dp_3 offload`. You can add
                        auto-wrap to `zero_dp_2` or with the same syntax:
                        zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.
                        (default: )
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json
                        config file (e.g. ds_config.json) or an already loaded
                        json file as a dict (default: None)
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no
                        label smoothing). (default: 0.0)
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor.
                        (default: False)
  --group_by_length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same
                        length together when batching. (default: False)
  --length_column_name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when
                        grouping by length. (default: length)
  --report_to REPORT_TO [REPORT_TO ...]
                        The list of integrations to report the results and
                        logs to. (default: None)
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag
                        `find_unused_parameters` passed to
                        `DistributedDataParallel`. (default: None)
  --no_dataloader_pin_memory
                        Whether or not to pin memory for DataLoader. (default:
                        True)
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader. (default:
                        True)
  --no_skip_memory_metrics
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: True)
  --skip_memory_metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: True)
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in
                        the Trainer. (default: False)
  --push_to_hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the
                        model hub after training. (default: False)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your
                        model. (default: None)
  --push_to_hub_model_id PUSH_TO_HUB_MODEL_ID
                        The name of the repository to which push the
                        `Trainer`. (default: None)
  --push_to_hub_organization PUSH_TO_HUB_ORGANIZATION
                        The name of the organization in with to which push the
                        `Trainer`. (default: None)
  --push_to_hub_token PUSH_TO_HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
  --mp_parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific
                        args. Ignored in Trainer (default: )
```