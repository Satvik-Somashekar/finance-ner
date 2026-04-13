#!/usr/bin/env python3
# Modified by Satvik Somashekar - Python 3 port + finance NER support

import os
import time
import codecs
import argparse          # CHANGED: optparse is deprecated in Python 3
import json
import numpy as np
from loader import prepare_sentence
from utils import create_input, iobes_iob, iob_ranges, zero_digits
from model import Model

# CHANGED: argparse instead of optparse
parser = argparse.ArgumentParser(description="NER Tagger - Finance Domain")
parser.add_argument("-m", "--model", default="", help="Model location")
parser.add_argument("-i", "--input", default="", help="Input file location")
parser.add_argument("-o", "--output", default="", help="Output file location")
parser.add_argument("-d", "--delimiter", default="__", help="Delimiter to separate words from their tags")
parser.add_argument("--outputFormat", default="", help="Output file format (json or default)")
opts = parser.parse_args()

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isfile(opts.input)

# Load existing model
print("Loading model...")          # CHANGED: print() with parentheses
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()

print('Tagging...')                # CHANGED: print() with parentheses
with codecs.open(opts.input, 'r', 'utf-8') as f_input:
    count = 0
    for line in f_input:
        words_ini = line.rstrip().split()
        if line:
            if parameters['lower']:
                line = line.lower()
            if parameters['zeros']:
                line = zero_digits(line)
            words = line.rstrip().split()

            sentence = prepare_sentence(words, word_to_id, char_to_id,
                                        lower=parameters['lower'])
            input = create_input(sentence, parameters, False)

            if parameters['crf']:
                y_preds = np.array(f_eval(*input))[1:-1]
            else:
                y_preds = f_eval(*input).argmax(axis=1)
            y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]

            if parameters['tag_scheme'] == 'iobes':
                y_preds = iobes_iob(y_preds)

            assert len(y_preds) == len(words)
            if opts.outputFormat == 'json':
                f_output.write(json.dumps({
                    "text": ' '.join(words),
                    "ranges": iob_ranges(y_preds)
                }))
            else:
                f_output.write('%s\n' % ' '.join(
                    '%s%s%s' % (w, opts.delimiter, y)
                    for w, y in zip(words_ini, y_preds)
                ))
        else:
            f_output.write('\n')
        count += 1
        if count % 100 == 0:
            print(count)           # CHANGED: print() with parentheses

print('---- %i lines tagged in %.4fs ----' % (count, time.time() - start))  # CHANGED
f_output.close()
