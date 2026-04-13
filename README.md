# Finance NER — Named Entity Recognition for Financial Text

A Named Entity Recognition (NER) system adapted for financial text, built on the 
BiLSTM-CRF architecture by Lample et al. (2016).

> Original paper: https://arxiv.org/abs/1603.01360  
> Original repo: https://github.com/glample/tagger

## Modifications
- Ported entire codebase from Python 2.7 to Python 3
- Replaced deprecated `optparse` with `argparse`
- Replaced `cPickle` with `pickle`
- Fixed all `print` statements for Python 3 compatibility
- Replaced `xrange` with `range`
- Applied to financial domain using the FiNER-ORD dataset

## Dataset
Uses the **FiNER-ORD** financial NER dataset sourced from SEC filings and financial news.  
Source: https://github.com/gtfintechlab/FiNER-ORD

## Architecture
Input → Character LSTM → Word Embeddings → BiLSTM → CRF → NER Tags

## Setup
```bash
pip install numpy theano
```

## Tag sentences
```bash
python3 tagger.py --model models/english/ --input input.txt --output output.txt
```

## Train on financial data
```bash
python3 train.py --train dataset/train.txt --dev dataset/dev.txt --test dataset/test.txt
```

## Author
Satvik Somashekar — VIT Vellore
