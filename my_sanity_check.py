#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    my_sanity_check.py 1h
    my_sanity_check.py 1i

"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT

import torch
import torch.nn as nn
import torch.nn.utils

from highway import Highway
from cnn import CNN

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1h_sanity_check():
    # Sanity check for highway.py 
    print ("-"*80)
    print("Running Sanity Check for Question 1h: Highway")
    print ("-"*80)

    inpt = torch.zeros(BATCH_SIZE, EMBED_SIZE)
    highway_net = Highway(EMBED_SIZE)
    output = highway_net.forward(inpt)
    output_expected_size = [BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1h: Highway!")
    print("-"*80)


def question_1i_sanity_check():
    # Sanity check for cnn.py 
    print ("-"*80)
    print("Running Sanity Check for Question 1i: CNN")
    print ("-"*80)

    m_word = 6
    e_char = 3
    e_word = 4
    inpt = torch.ones(BATCH_SIZE, e_char, m_word)
    cnn_net = CNN(e_char, e_word)
    output = cnn_net.forward(inpt)
    output_expected_size = [BATCH_SIZE, e_word]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1i: CNN!")
    print("-"*80)


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    if args['1h']:
        question_1h_sanity_check()
    elif args['1i']:
        question_1i_sanity_check()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
