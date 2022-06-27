![HUPD-Diagram](https://github.com/suzgunmirac/hupd/blob/main/legacy/figures/HUPD-Logo.png)

# The Harvard USPTO Patent Dataset (HUPD)
This present repository contains the dataset from "[_The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications_](https://openreview.net/pdf?id=WhTTCWsMrYv)", which is currently under review in the NeurIPS 2022 Datasets and Benchmarks Track.

## Table of Contents
1. [Overview of HUPD](#overview-of-hupd)
2. [Usage: Loading the Dataset](#usage)
3. [Downloading the Dataset](#downloading-the-dataset)
4. [Data Fields and Data Format](#data-fields-and-data-format)
5. [Google Colab](#google-colab)
6. [Jupyter Notebooks](#jupyter-notebooks)
7. [Experiments and Tasks](#experiments-and-tasks)
8. [Citation](#citation)
9. [Licensing and Disclaimer](#licensing-and-contact)

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
HUPD is also available on Google Drive. This Google Drive folder contains four large tarred files and a big feather file. **More than 360GB of disk storage space** is needed to download and store all the individual files. The following command will download all the tar files and then extract them. 

```bash
bash ./scripts/download_and_extract_all.sh
```

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
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_ZsI7WFTsEO0iu_0g3BLTkIkOUqPzCET?usp=sharing) [HUPD Examples: Loading the Dataset](https://colab.research.google.com/drive/1_ZsI7WFTsEO0iu_0g3BLTkIkOUqPzCET?usp=sharing)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TzDDCDt368cUErH86Zc_P2aw9bXaaZy1?usp=sharing) [HUPD Examples: Loading HUPD By Using HuggingFace's Libraries](https://colab.research.google.com/drive/1TzDDCDt368cUErH86Zc_P2aw9bXaaZy1?usp=sharing)

## Jupyter Notebooks 
Please feel free to take a look at our notebooks if you would like to run the code in an interactive session or plot some of the figures in our paper by yourself.
* `Exploring the Data Fields of HUPD.ipynb`: To explore some of the data fields within HUPD.
* `Loading HUPD By Using HuggingFace's Libraries.ipynb`: To learn how to load and use HUPD using Hugging Face's libraries. 


## Experiments and Tasks
Let us first provide a brief overview of each task we consider in our paper:
- **Patent Acceptance Prediction**: Given a section of a patent application (in particular, the asbtract, claims, or description), we predict whether the application will be accepted by the USPTO.
- **Automated Subject (IPC/CPC) Classification**: We predict the primary IPC or CPC code  of a patent application given (some subset of) the text of the application.
- **Language Modeling**: We perform masked language modeling on the claims and description sections of patent applications.
- **Abstractive Summarization**: Each patent contains an abstract section in which the applicant summarizes the content of the patent. We use this section as the ground truth for our abstractive summarization task, and we use either the claims section or the description section as the source text.

### Model Weights
The model weights can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/12c6tIsaKisTR-ujukGXjbllk6gRGFRvx?usp=sharing).

## Citation
If your research makes use of our dataset, models, or results, please consider citing our paper. 
```
@article{suzgun2022hupd,
  title={The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications},
  author={Suzgun, Mirac and Melas-Kyriazi, Luke and Sarkar, Suproteem K and Kominers, Scott and Shieber, Stuart},
  year={2022}
}
```

## Licensing and Contact
- The Harvard USPTO Dataset (HUPD) is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License, while the codes and the pretrained models in this repository are under the MIT License. 
- Contact msuzgun@cs.stanford.edu with any questions, comments, or suggestions.