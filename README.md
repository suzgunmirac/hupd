![HUPD-Diagram](https://github.com/suzgunmirac/hupd/blob/main/legacy/figures/HUPD-Logo.png)

# The Harvard USPTO Patent Dataset (HUPD) [![arXiv](https://img.shields.io/badge/arXiv-2207.04043-b31b1b.svg)](https://arxiv.org/abs/2207.04043)
This present repository contains the dataset from "[The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications](https://arxiv.org/abs/2207.04043)", which has recently been accepted to the NeurIPS 2023 Datasets and Benchmarks Track.

**N.B.** We will be updating our GitHub repository and website shortly.

## Table of Contents
1. [Overview of HUPD](#overview-of-hupd)
2. [Usage: Loading the Dataset](#usage)
3. [Downloading the Dataset](#downloading-the-dataset)
4. [Data Fields and Data Format](#data-fields-and-data-format)
5. [Google Colab](#google-colab)
6. [Experiments and Tasks](#experiments-and-tasks)
7. [Citation](#citation)
8. [Licensing and Contact](#licensing-and-contact)

## Overview of HUPD
The Harvard USPTO Dataset (HUPD) is a large-scale, well-structured, and multi-purpose corpus of English-language utility patent applications filed to the United States Patent and Trademark Office (USPTO) between January 2004 and December 2018. With more than 4.5 million patent documents, HUPD is two to three times larger than comparable patent datasets. Unlike previously proposed patent datasets in NLP, it contains the inventor-submitted versions of patent applications, not the final versions of granted patents, allowing us to study patentability at the time of filing using NLP methods for the first time. It is also novel in its inclusion of rich structured metadata alongside the text of patent filings: By providing each application's metadata along with all of its text fields, the dataset enables researchers to perform new sets of NLP tasks that leverage variation in structured covariates.


As a case study on the types of research HUPD makes possible, we introduce a new task to the NLP community, namely patent acceptance prediction. We additionally show the structured metadata provided in the dataset allows us to conduct explicit studies of concept shifts for this task. Finally, we demonstrate how our dataset can be used for three additional tasks: Multi-class classification of patent subject areas, language modeling, and summarization. Overall, HUPD is one of the largest multi-purpose NLP datasets containing domain-specific textual data, along with well-structured bibliographic metadata, and aims to advance research extending language and classification models to diverse and dynamic real-world data distributions.


## Usage
### Loading the Dataset
#### Sample (January 2016 Subset) 
The following command can be used to load the `sample` version of the dataset, which contains all the patent applications that were filed to the USPTO during the month of January in 2016. This small subset of the dataset can be used for debugging and exploration purposes.

```python
from datasets import load_dataset

dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
)
```

#### Full Dataset

If you would like to use the **full** version of the dataset, please make sure that change the `name` field from `sample` to `all`, specify the training and validation start and end dates carefilly, and set `force_extract` to be `True` (so that you would only untar the files that you are interested in and not squander your disk storage space). In the following example, for instance, we set the training set year range to be [2011, 2016] (inclusive) and the validation set year range to be 2017.

```python
from datasets import load_dataset

dataset_dict = load_dataset('HUPD/hupd',
    name='all',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
    icpr_label=None,
    force_extract=True,
    train_filing_start_date='2011-01-01',
    train_filing_end_date='2016-12-31',
    val_filing_start_date='2017-01-01',
    val_filing_end_date='2017-12-31',
)
```

## Downloading the Dataset 
### Manual Download Options:

#### Hugging Face Datasets
HUPD can be easily accessed through Hugging Face Datasets. To download the raw patent application files in HUPD, please go to [this link](https://huggingface.co/datasets/HUPD/hupd/blob/main/data/all-years.tar), uncompress the `all-years.tar` file, and then further uncompress the new `[year].tar` files that are of interest to you. 


#### Google Drive
HUPD is also available on Google Drive. This Google Drive folder contains four large tarred files and a big feather file. **More than 360GB of disk storage space** is needed to download and store all the individual files.

## Data Fields and Data Format
Each patent application is defined by a distinct JSON file, named after its application number, and includes information about the application and publication numbers, title, decision status, filing and publication dates, primary and secondary classification codes, inventor(s), examiner, attorney, abstract, claims, background, summary, and full description of the proposed invention, among other fields. There are also supplementary variables, such as the small-entity indicator (which denotes whether the applicant is considered to be a small entity by the USPTO) and the foreign-filing indicator (which denotes whether the application was originally filed in a foreign country). 

- In total, there are 34 data fields for each application:
```python
{
    "application_number": "...",
    "publication_number": "...",
    "title": "...",
    "decision": "...",
    "date_produced": "...",
    "date_published": "...",
    "main_cpc_label": "...",
    "cpc_labels": ["...", "...", "..."],
    "main_ipcr_label": "...",
    "ipcr_labels": ["...", "...", "..."],
    "patent_number": "...",
    "filing_date": "...",
    "patent_issue_date": "...",
    "abandon_date": "...",
    "uspc_class": "...",
    "uspc_subclass": "...",
    "examiner_id": "...",
    "examiner_name_last": "...",
    "examiner_name_first": "...",
    "examiner_name_middle": "...",
    "inventor_list": [
        {
            "inventor_name_last": "...",
            "inventor_name_first": "...",
            "inventor_city": "...",
            "inventor_state": "...",
            "inventor_country": "..."
        }
    ],
    "abstract": "...",
    "claims": "...",
    "background": "...",
    "summary": "...",
    "full_description": "..."
}
```

## Google Colab
You can also use the following Google Colab notebooks to explore HUPD. 
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_ZsI7WFTsEO0iu_0g3BLTkIkOUqPzCET?usp=sharing) [ HUPD Examples: Loading the Dataset](https://colab.research.google.com/drive/1_ZsI7WFTsEO0iu_0g3BLTkIkOUqPzCET?usp=sharing)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TzDDCDt368cUErH86Zc_P2aw9bXaaZy1?usp=sharing) [ HUPD Examples: Loading HUPD By Using HuggingFace's Libraries](https://colab.research.google.com/drive/1TzDDCDt368cUErH86Zc_P2aw9bXaaZy1?usp=sharing)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TzDDCDt368cUErH86Zc_P2aw9bXaaZy1?usp=sharing) [ HUPD Examples: Using the HUPD DistilRoBERTa Model](https://colab.research.google.com/drive/11t69BWcAVXndQxAOCpKaGkKkEYJSfydT?usp=sharing)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TzDDCDt368cUErH86Zc_P2aw9bXaaZy1?usp=sharing) [ HUPD Examples: Using the HUPD T5-Small Summarization Model](https://colab.research.google.com/drive/1VkCtrRIryzev_ixDjmJcfJNK-q6Vx24y?usp=sharing)

## Experiments and Tasks
Let us first provide a brief overview of each task we consider in our paper:
- **Patent Acceptance Prediction**: Given a section of a patent application (in particular, the asbtract, claims, or description), we predict whether the application will be accepted by the USPTO.
- **Automated Subject (IPC/CPC) Classification**: We predict the primary IPC or CPC code  of a patent application given (some subset of) the text of the application.
- **Language Modeling**: We perform masked language modeling on the claims and description sections of patent applications.
- **Abstractive Summarization**: Each patent contains an abstract section in which the applicant summarizes the content of the patent. We use this section as the ground truth for our abstractive summarization task, and we use either the claims section or the description section as the source text.

## Models

### HUPD DistilRoBERTa-Base Masked Language Model
[HUPD DistilRoBERTa-Base](https://huggingface.co/turingmachine/hupd-distilroberta-base) was fine-tuned on HUPD with a masked language modeling objective. You can use this model directly with the Hugging Face pipeline as follows:

```python
from transformers import pipeline

model = pipeline(task="fill-mask", model="turingmachine/hupd-distilroberta-base")
model("Improved <mask> for playing a game of thumb wrestling.")
```

Here is the output:
```python
[{'score': 0.4274042248725891,
  'sequence': 'Improved method for playing a game of thumb wrestling.',
  'token': 5448,
  'token_str': ' method'},
 {'score': 0.06967400759458542,
  'sequence': 'Improved system for playing a game of thumb wrestling.',
  'token': 467,
  'token_str': ' system'},
 {'score': 0.06849079579114914,
  'sequence': 'Improved device for playing a game of thumb wrestling.',
  'token': 2187,
  'token_str': ' device'},
 {'score': 0.04544765502214432,
  'sequence': 'Improved apparatus for playing a game of thumb wrestling.',
  'token': 26529,
  'token_str': ' apparatus'},
 {'score': 0.025765646249055862,
  'sequence': 'Improved means for playing a game of thumb wrestling.',
  'token': 839,
  'token_str': ' means'}]
```

### HUPD T5-Small Summarization Model
[HUPD T5-Small](https://huggingface.co/turingmachine/hupd-t5-small) was fine-tuned on the claims (text) and abstract (summary) sections of HUPD. You can use this model directly with the Hugging Face pipeline as follows:

```python
from transformers import pipeline

TEXT = "1. An optical coherent receiver for an optical communication network, said optical coherent receiver being configured to receive a modulated optical signal and to process said modulated optical signal for generating an in-phase component and a quadrature component, said in-phase component and said quadrature component being electrical signals, said optical coherent receiver comprising a power adjuster in turn comprising: a multiplying unit configured to multiply said in-phase component by an in-phase gain thereby providing a power-adjusted in-phase component, and to multiply said quadrature component by a quadrature gain thereby providing a power-adjusted quadrature component; and a digital circuit connected between output and input of said multiplying unit and configured to compute: a common gain indicative of a sum of a power of said power-adjusted in-phase component and a power of said power-adjusted quadrature component, and a differential gain indicative of a difference between said power of said power-adjusted in-phase component and said power of said power-adjusted quadrature component; and said in-phase gain as a product between said common gain and said differential gain, and said quadrature gain as a ratio between said common gain and said differential gain. 2. An optical coherent receiver according to claim 1, wherein it further comprises an analog-to-digital unit connected at the input of said power adjuster, said analog-to-digital unit being configured to ..."

summarizer = pipeline(task="summarization", model="turingmachine/hupd-t5-small")
summarizer(TEXT)
```

Here is the output:
```python
[{'summary_text': 'An optical coherent receiver for an optical communication network includes a power adjuster and a digital circuit connected between output and input of the multiplying unit and configured to compute a common gain indicative of a sum of the power of an in-phase component and the power-adjusted quadrature component, and the differential gain as a product between the common gain and the diffractive gain.'}]
```

### Model Weights
The model weights can also be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/12c6tIsaKisTR-ujukGXjbllk6gRGFRvx?usp=sharing).

## Citation
If your research makes use of our dataset, models, or results, please consider citing our paper. 
```
@inproceedings{
suzgun2023the,
    title={The Harvard {USPTO} Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications},
    author={Mirac Suzgun and Luke Melas-Kyriazi and Suproteem K Sarkar and Scott Kominers and Stuart Shieber},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=tk27oD2cBw}
}
```

## Licensing and Contact
HUPD is released under the Creative Commons Attribution 4.0 International License. If you have any questions, comments, or suggestions, please feel free to reach out to msuzgun@stanford.edu.
