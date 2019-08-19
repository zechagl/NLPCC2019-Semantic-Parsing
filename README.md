# NLPCC2019-Semantic-Parsing

NLPCC19: A Sketch-Based System for Semantic Parsing

## Setup

### Requirements

- Python 3.6
- Tensorflow 1.11 (GPU)

### Install Python dependency

```sh
pip install -r requirements.txt
```

### Download data and pretrained models

Download Bert pretrained model [BERT-Base, Uncased](https://github.com/google-research/bert)

Download the data file from [Google Drive](https://drive.google.com/open?id=19faRIaxT-z9rA2CSD7er1WCDnTJHLAU6)

Download pretrained models from [Google Drive](https://drive.google.com/open?id=1ocmWJhCDLt5S8TEtHTPemd0mbUKxs72I)

## Usage

### Run multi-task model

```sh
sh multitask.sh
```

### Run pattern pair matching net

```sh
sh pattern_pair.sh
```

### Run predicate-entity pair matching net

```sh
sh pep.sh
```

### Run Pointer net

```sh
sh pointer.sh
```

### Rank by 3 intermediate scores

```sh
sh score.sh
```

## Contact information

Any questions : zcli18@pku.edu.cn
