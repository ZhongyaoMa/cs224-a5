#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx = pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.e_char = 50
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx = pad_token_idx)
        self.embed_size = embed_size
        self.cnn = CNN(self.e_char, self.embed_size)
        self.highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(p = 0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        sents_len, batch_size, m_word = input.shape
        x_emb = self.embeddings(input)              # (sentence_length, batch_size, max_word_length, e_char)

        x_convout = self.cnn(x_emb.view(-1, m_word, self.e_char).permute(0, 2, 1))    # (sentence_length * batch_size, e_word)
        x_highway = self.highway(x_convout)         # (sentence_length * batch_size, e_word)
        x_word_emb = self.dropout(x_highway)        # (sentence_length * batch_size, e_word)

        return x_word_emb.view(sents_len, batch_size, self.embed_size)   # (sentence_length, batch_size, e_word)

        ### END YOUR CODE

