#!/usr/bin/env python3
# Modified by Satvik Somashekar - Python 3 port

import os
import numpy as np
import argparse                          # CHANGED: optparse → argparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader
from utils import models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model

# Read parameters from command line
parser = argparse.ArgumentParser(description="NER Tagger Trainer")  # CHANGED
parser.add_argument("-T", "--train", default="", help="Train set location")
parser.add_argument("-d", "--dev", default="", help="Dev set location")
parser.add_argument("-t", "--test", default="", help="Test set location")
parser.add_argument("-s", "--tag_scheme", default="iobes", help="Tagging scheme (IOB or IOBES)")
parser.add_argument("-l", "--lower", default=0, type=int, help="Lowercase words")
parser.add_argument("-z", "--zeros", default=0, type=int, help="Replace digits with 0")
parser.add_argument("-c", "--char_dim", default=25, type=int, help="Char embedding dimension")
parser.add_argument("-C", "--char_lstm_dim", default=25, type=int, help="Char LSTM hidden layer size")
parser.add_argument("-b", "--char_bidirect", default=1, type=int, help="Use bidirectional LSTM for chars")
parser.add_argument("-w", "--word_dim", default=100, type=int, help="Token embedding dimension")
parser.add_argument("-W", "--word_lstm_dim", default=100, type=int, help="Token LSTM hidden layer size")
parser.add_argument("-B", "--word_bidirect", default=1, type=int, help="Use bidirectional LSTM for words")
parser.add_argument("-p", "--pre_emb", default="", help="Location of pretrained embeddings")
parser.add_argument("-A", "--all_emb", default=0, type=int, help="Load all embeddings")
parser.add_argument("-a", "--cap_dim", default=0, type=int, help="Capitalization feature dimension")
parser.add_argument("-f", "--crf", default=1, type=int, help="Use CRF (0 to disable)")
parser.add_argument("-D", "--dropout", default=0.5, type=float, help="Dropout on the input")
parser.add_argument("-L", "--lr_method", default="sgd-lr_.005", help="Learning method")
parser.add_argument("-r", "--reload", default=0, type=int, help="Reload the last saved model")
opts = parser.parse_args()

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=parameters, models_path=models_path)
print("Model location: %s" % model.model_path)   # CHANGED: print()

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Word mappings
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower)
dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower)
test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)

print("%i / %i / %i sentences in train / dev / test." % (   # CHANGED: print()
    len(train_data), len(dev_data), len(test_data)))

print('Saving the mappings to disk...')                       # CHANGED: print()
model.save_mappings(id_to_word, id_to_char, id_to_tag)

f_train, f_eval = model.build(**parameters)

if opts.reload:
    print('Reloading previous model...')                      # CHANGED: print()
    model.reload()

# Train network
singletons = set([word_to_id[k] for k, v in dico_words_train.items() if v == 1])
n_epochs = 100
freq_eval = 1000
best_dev = -np.inf
best_test = -np.inf
count = 0

for epoch in range(n_epochs):          # CHANGED: xrange → range (xrange removed in Python 3)
    epoch_costs = []
    print("Starting epoch %i..." % epoch)                    # CHANGED: print()
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        input = create_input(train_data[index], parameters, True, singletons)
        new_cost = f_train(*input)
        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0:
            print("%i, cost average: %f" % (i, np.mean(epoch_costs[-50:])))  # CHANGED
        if count % freq_eval == 0:
            dev_score = evaluate(parameters, f_eval, dev_sentences,
                                 dev_data, id_to_tag, dico_tags)
            test_score = evaluate(parameters, f_eval, test_sentences,
                                  test_data, id_to_tag, dico_tags)
            print("Score on dev: %.5f" % dev_score)          # CHANGED: print()
            print("Score on test: %.5f" % test_score)        # CHANGED: print()
            if dev_score > best_dev:
                best_dev = dev_score
                print("New best score on dev.")               # CHANGED: print()
                print("Saving model to disk...")              # CHANGED: print()
                model.save()
            if test_score > best_test:
                best_test = test_score
                print("New best score on test.")              # CHANGED: print()
    print("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))  # CHANGED
