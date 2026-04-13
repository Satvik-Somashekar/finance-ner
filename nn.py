#!/usr/bin/env python3
# Modified by Satvik Somashekar - Python 3 port

import theano
import theano.tensor as T
from utils import shared


class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation='sigmoid',
                 name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        self.weights = shared((input_dim, output_dim), name + '__weights')
        self.bias = shared((output_dim,), name + '__bias')

        if self.bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def link(self, input):
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """
    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        self.embeddings = shared((input_dim, output_dim),
                                 self.name + '__embeddings')
        self.params = [self.embeddings]

    def link(self, input):
        self.input = input
        self.output = self.embeddings[self.input]
        return self.output


class DropoutLayer(object):
    """
    Dropout layer. Randomly set to 0 values of the input
    with probability p.
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        assert 0. <= p < 1.
        self.p = p
        self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
        self.name = name

    def link(self, input):
        if self.p > 0:
            mask = self.rng.binomial(n=1, p=1-self.p, size=input.shape,
                                     dtype=theano.config.floatX)
            self.output = input * mask
        else:
            self.output = input
        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        self.w_xi = shared((input_dim, hidden_dim), name + '__w_xi')
        self.w_hi = shared((hidden_dim, hidden_dim), name + '__w_hi')
        self.w_ci = shared((hidden_dim, hidden_dim), name + '__w_ci')

        self.w_xo = shared((input_dim, hidden_dim), name + '__w_xo')
        self.w_ho = shared((hidden_dim, hidden_dim), name + '__w_ho')
        self.w_co = shared((hidden_dim, hidden_dim), name + '__w_co')

        self.w_xc = shared((input_dim, hidden_dim), name + '__w_xc')
        self.w_hc = shared((hidden_dim, hidden_dim), name + '__w_hc')

        self.b_i = shared((hidden_dim,), name + '__b_i')
        self.b_c = shared((hidden_dim,), name + '__b_c')
        self.b_o = shared((hidden_dim,), name + '__b_o')
        self.c_0 = shared((hidden_dim,), name + '__c_0')
        self.h_0 = shared((hidden_dim,), name + '__h_0')

        self.params = [self.w_xi, self.w_hi, self.w_ci,
                       self.w_xo, self.w_ho, self.w_co,
                       self.w_xc, self.w_hc,
                       self.b_i, self.b_c, self.b_o,
                       self.c_0, self.h_0]

    def link(self, input):
        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) +
                                 T.dot(h_tm1, self.w_hi) +
                                 T.dot(c_tm1, self.w_ci) +
                                 self.b_i)
            c_t = ((1 - i_t) * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) +
                   T.dot(h_tm1, self.w_hc) + self.b_c))
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) +
                                 T.dot(h_tm1, self.w_ho) +
                                 T.dot(c_t, self.w_co) +
                                 self.b_o)
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim)
                            for x in [self.c_0, self.h_0]]
        else:
            self.input = input
            outputs_info = [self.c_0, self.h_0]

        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]
        return self.output


def log_sum_exp(x, axis=None):
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)
