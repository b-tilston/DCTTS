# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#tf.keras.mixed_precision.experimental.Policy('mixed_float16')

# Inserted a layer count next to each layer to make the architecture of the model readily apparent
def TextEnc(L, training=True): #15 layers in total
    '''
    Args:
      L: Text inputs. (B, N)

    Return:
        K: Keys. (B, N, d)
        V: Values. (B, N, d)
    '''
    i = 1
    tensor = embed(L,
                   vocab_size=len(hp.vocab),
                   num_units=hp.e,
                   scope="embed_{}".format(i)); i += 1 #Text2Mel/TextEnc/embed_1/
    tensor = conv1d(tensor,
                    filters= 2*hp.d,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/TextEnc/C_2/
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/TextEnc/C_3/

    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            activation_fn=None,
                            training=training,
                            scope="HC_{}".format(i)); i += 1 #Text2Mel/TextEnc/HC_4/ - Text2Mel/TextEnc/HC_11/ 
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1 #Text2Mel/TextEnc/HC_12/ + Text2Mel/TextEnc/HC_13/

    for _ in range(2):
        tensor = hc(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1 #Text2Mel/TextEnc/HC_14/ + Text2Mel/TextEnc/HC_15/

    K, V = tf.split(tensor, 2, -1)
    return K, V

def AudioEnc(S, training=True): #13 layers in total
    '''
    Args:
      S: melspectrogram. (B, T/r, n_mels)

    Returns
      Q: Queries. (B, T/r, d)
    '''
    i = 1
    tensor = conv1d(S,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/AudioEnc/C_1/
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/AudioEnc/C_2/
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/AudioEnc/C_3/
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            padding="CAUSAL",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1 #Text2Mel/AudioEnc/HC_4/ - Text2Mel/AudioEnc/HC_11/
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=3,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1 #Text2Mel/AudioEnc/HC_12/ + Text2Mel/AudioEnc/HC_13/

    return tensor

def Attention(Q, K, V, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)
      mononotic_attention: A boolean. At training, it is False.
      prev_max_attentions: (B,). At training, it is set to None.

    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      alignments: (B, N, T/r)
      max_attentions: (B, T/r)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.math.rsqrt(tf.compat.v1.to_float(hp.d)) #tf.to_float(hp.d)
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions, hp.max_N)
        reverse_masks = tf.sequence_mask(hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)
    A = tf.nn.softmax(A) # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)

    return R, alignments, max_attentions

def AudioDec(R, training=True):
    '''
    Args:
      R: [Context Vectors; Q]. (B, T/r, 2d)

    Returns:
      Y: Melspectrogram predictions. (B, T/r, n_mels)
    '''

    i = 1
    tensor = conv1d(R,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/AudioDec/C_1/
    for j in range(4):
        tensor = hc(tensor,
                        size=3,
                        rate=3**j,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1 #Text2Mel/AudioDec/HC_2/ + Text2Mel/AudioDec/hC_3/ + #Text2Mel/AudioDec/hC_4/ + #Text2Mel/AudioDec/hC_5/

    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1 #Text2Mel/AudioDec/HC_6/ + Text2Mel/AudioDec/HC_7/
    for _ in range(3):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1 #Text2Mel/AudioDec/C_8/ + Text2Mel/AudioDec/C_9/ + Text2Mel/AudioDec/C_10/
    # mel_hats
    logits = conv1d(tensor,
                    filters=hp.n_mels,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #Text2Mel/AudioDec/C_11/
    Y = tf.nn.sigmoid(logits) # mel_hats

    return logits, Y

def SSRN(Y, training=True):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)

    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''

    i = 1 # number of layers

    # -> (B, T/r, c)
    tensor = conv1d(Y,
                    filters=hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #SSRN/C_1/
    for j in range(2):
        tensor = hc(tensor,
                      size=3,
                      rate=3**j,
                      dropout_rate=hp.dropout_rate,
                      training=training,
                      scope="HC_{}".format(i)); i += 1 #SSRN/HC_2 + SSRN/HC_3/
    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor,
                                  scope="D_{}".format(i),
                                  dropout_rate=hp.dropout_rate,
                                  training=training,); i += 1 #SSRN/D_4/ + SSRN/D_7/
        for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1 #SSRN/HC_5/ + SSRN/HC_6/ + SSRN/HC_8/ + SSRN/HC_9/
    # -> (B, T, 2*c)
    tensor = conv1d(tensor,
                    filters=2*hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #SSRN/C_10/
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1 #SSRN/HC_11/ + SSRN/HC_12/
    # -> (B, T, 1+n_fft/2)
    tensor = conv1d(tensor,
                    filters=1+hp.n_fft//2,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1 #SSRN/C_13/

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1 ##SSRN/C_14/ + SSRN/C_15/
    logits = conv1d(tensor,
               size=1,
               rate=1,
               dropout_rate=hp.dropout_rate,
               training=training,
               scope="C_{}".format(i)) #SSRN/C_16/
    Z = tf.nn.sigmoid(logits) 
    return logits, Z

