#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: May 5 2020
# Last Modified: Mar 9 2023
#

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from keras import activations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import functools
import math
import json

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

DEFAULT_PARTITION_CONFIG = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":False
}

EDGETPU_BERT_TINY = EDGETPU_BERT_MINI = EDGETPU_BERT_MEDIUM = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":True
}

EDGETPU_BERT_BASE = {
    "intermediate_partitions":2,
    "fc_out_partitions":2,
    "embedding_partitions":2,
    "use_conv":True
}

EDGETPU_BERT_LARGE = {
    "intermediate_partitions":2,
    "fc_out_partitions":2,
    "embedding_partitions":2,
    "use_conv":True
}

def approx_gelu(x):
    return x*tf.math.sigmoid(1.702*x)

class ScaledDotProduct(tf.keras.layers.Layer):
    def __init__(self, inp_size, d_model, activation='gelu', partition_config=None, name=None):
        super(ScaledDotProduct, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG
        self.cfg_json = json.dumps(partition_config)
        self.partition_config=partition_config
        self.d_model = d_model
        self.activation = activation
        self.inp_size = inp_size
        self.query_activation = activations.get(activation)
        self.key_activation = activations.get(activation)
        self.value_activation = activations.get(activation)
        self.softmax = activations.get('softmax')
        use_conv = partition_config["use_conv"]
        self.use_conv = use_conv

        self.dense_q = ConfigurableDense(self.d_model, inp_size=self.inp_size, name="query", use_conv=use_conv)
        self.dense_k = ConfigurableDense(self.d_model, inp_size=self.inp_size, name="key", use_conv=use_conv)
        self.dense_v = ConfigurableDense(self.d_model, inp_size=self.inp_size, name="value", use_conv=use_conv)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.inp_size)

    def build(self, input_shape):
        pass

    def get_config(self):
        return {
            'inp_size':self.inp_size,
            'd_model': self.d_model,
            'name':self.name,
            'activation':self.activation,
        }


    def call(self, q, k, v, mask):
        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        #if not mask is None:
        scaled_attention_logits += mask
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return  output

class BERTMultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, rate=0.1, partition_config=None, activation='gelu', name=None):
        super(BERTMultiHeadedAttention, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG
        self.partition_config = partition_config
        self.attention_heads = []
        self.use_conv = partition_config["use_conv"]
        for i in range(num_heads):
            sdp = ScaledDotProduct(d_model, int(d_model/num_heads), partition_config=partition_config, name=self.name + 'sdp_' + str(i))
            self.attention_heads.append(sdp)
        self.num_heads = num_heads
        self.rate = rate
        self.activation = activation
        self.act_out = activations.get(activation)
        self.d_model = d_model
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.mha_ffn = ConfigurableDense(self.d_model, inp_size=self.d_model, use_conv=self.use_conv, name=self.name + "attention_output")

    def build(self, input_shape):
        pass

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'name':self.name,
            'activation':self.activation,
            'rate':self.rate,
            'partition_config':self.partition_config
            }

    def call(self, q, k, v, mask, training=False):
        attention_outputs = []
        for head in self.attention_heads:
            attention_outputs.append(head(q, k, v, mask))
        x = attention_outputs[0]
        if self.num_heads>1:
            x = tf.keras.layers.concatenate(attention_outputs)
        x = self.dropout1(x, training=training)
        x = self.mha_ffn(x)
        x = self.dropout2(x, training=training)
        return x

class ConfigurableDense(tf.keras.layers.Layer):
    def __init__(self, size, use_conv=False, inp_size=None, use_bias=True, activation=None, name=None):
        super(ConfigurableDense, self).__init__(name=name)
        self.size = size
        self.use_conv = use_conv
        self.use_bias = use_bias
        self.inp_size = inp_size
        self.activation = activation
        self.act_out = activations.get(activation)
        if activation=="gelu":
            self.act_out=approx_gelu

    def get_config(self):
        return {
            'name': self.name,
            'size': self.size,
            'use_conv': self.use_conv,
            'use_bias':self.use_bias,
            'inp_size': self.inp_size,
            'activation':self.activation,
            }

    def build(self, input_shape):
        inp_size = self.inp_size
        if inp_size is None:
            inp_size = input_shape[-1]
        if not self.use_conv:
            self.kernel = self.add_weight("kernel",shape=[self.inp_size,self.size],
                    initializer='random_normal',
                    trainable=True)
        else:
            kernel_shape = (1,inp_size,self.size)
            self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight("bias",shape=[self.size],
                initializer='random_normal',
                trainable=True)

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=list(conv_utils.normalize_tuple(1, 1, 'strides')),
            name='conv1d')

    
    def call(self, x):
        if not self.use_conv:
            out = tf.matmul(x, self.kernel)
        else:
            out = tf.expand_dims(x, axis=0)
            out = self._convolution_op(out, self.kernel)
            out = tf.squeeze(out,axis=0)
        if self.use_bias:
            out += self.bias
        if self.activation is not None:
            out = self.act_out(out)
        return out


class PartitionLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, input_size, num_layers=1, partition_output=True, rank=2, use_conv=False, use_bias=True, activation=None, name=None):
        super(PartitionLayer, self).__init__(name=name)
        if partition_output:
          assert output_size%num_layers==0
        else:
          assert input_size%num_layers==0
        self.num_layers = num_layers
        self.output_size = output_size
        self.rank = rank
        self.use_conv = use_conv
        self.input_size=input_size
        self.activation = activation
        self.use_bias = use_bias
        self.partition_output = partition_output
        self.fcs = []
        for i in range(num_layers):
          if partition_output:
            self.fcs.append(ConfigurableDense(int(output_size/num_layers), inp_size=input_size, activation=activation, use_conv=use_conv, name="partition_out" + str(i)))
          else:
            self.fcs.append(ConfigurableDense(output_size, inp_size=int(input_size/num_layers), activation=activation, use_conv=use_conv, use_bias=False, name=str(i)))

    def build(self, input_shape):
      if self.use_bias and not self.partition_output:
        self.bias = self.add_weight("bias",shape=[self.output_size],
            initializer='random_normal',
            trainable=True)

    def get_config(self):
        return {
            'name':self.name,
            'use_conv':self.use_conv
            }


    def call(self, x):
      if self.partition_output:
        outputs = []
        for i in range(self.num_layers):
          outputs.append(self.fcs[i](x))
        x = outputs[0]
        if self.num_layers>1:
            x = tf.keras.layers.concatenate(outputs)
        return x
      rank = self.rank
      partition_size = int(self.input_size/self.num_layers)
      if rank==2:
        output = self.fcs[0](x[:,0:partition_size])
        for i in range(1,self.num_layers):
          output+=self.fcs[i](x[:,partition_size*(i):partition_size*(i+1)])
        if self.use_bias:
            output+=self.bias
        return output
      elif rank==3:
        output = self.fcs[0](x[:,:,0:partition_size])
        for i in range(1,self.num_layers):
          output+=self.fcs[i](x[:,:,partition_size*(i):partition_size*(i+1)])
        if self.use_bias:
          output+=self.bias
        return output
      return x


class PartitionEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_Size, emb_size, n_partitions=1, name=None):
      super(PartitionEmbedding, self).__init__(name=name)
      self.vocab_size = vocab_Size
      self.emb_size = emb_size
      self.n_partitions = n_partitions
      self.partition_size = int(self.emb_size/n_partitions)
      assert self.emb_size % n_partitions == 0
      self.embeddings = [tf.keras.layers.Embedding(vocab_Size, self.partition_size, name="partition" + str(i)) for i in range(n_partitions)]

  def call(self, x):
      outputs = []
      for embedding in self.embeddings:
          outputs.append(embedding(x))
      x = outputs[0]
      if self.n_partitions>1:
          x = tf.keras.layers.concatenate(outputs)
      return x


class BertEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, seq_len, n_segments, d_model, n_partitions=1, name=None):
        super(BertEmbedding, self).__init__(name=name)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.d_model = d_model
        self.n_partitions = n_partitions

        self.word_embeddings = PartitionEmbedding(vocab_size, d_model, n_partitions=n_partitions, name="word_embeddings")
        self.position_embedding = PartitionEmbedding(seq_len, d_model, n_partitions=n_partitions, name="position_embeddings")
        self.type_embeddings = PartitionEmbedding(n_segments, d_model, n_partitions=n_partitions, name="type_embeddings")
        self.norm = DynamicLayerNormalization(epsilon=1e-12, name="emb_layer_normalization")

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_segments':self.n_segments,
            'd_model':self.d_model,
            'n_partitions':self.n_partitions,
            'name':self.name
            }

    def build(self, input_shape):
        pass

    def call(self, x, seg):
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        pos = tf.range(seq_len)
        pos = tf.expand_dims(pos, 0)
        pos = tf.broadcast_to(pos, (batch_size, seq_len))
        embedding = self.word_embeddings(x) + self.type_embeddings(seg) + self.position_embedding(pos)
        return self.norm(embedding)

class BERT(tf.keras.layers.Layer):
    def __init__(self, n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size, rate=0.1, partition_config=None, activation='gelu', name=None):
        super(BERT, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG
        self.partition_config=partition_config
        self.rate = rate
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.use_conv = partition_config["use_conv"]
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.n_partitions = partition_config["intermediate_partitions"]

        embedding_partitions = partition_config["embedding_partitions"]


        self.embedding = BertEmbedding(vocab_size, seq_len, n_segments, d_model, n_partitions=embedding_partitions)
        self.enc_layers = [BertEncoder(num_heads, d_model, intermediate_size, rate=rate, activation=activation, partition_config=partition_config, name="layer_" + str(i)) 
                        for i in range(n_layers)]

        self.activation = activation
        self.act_out = activations.get(activation)
        if activation == 'gelu':
            self.act_out = approx_gelu

        self.pooler_ffn = ConfigurableDense(self.d_model, inp_size=self.d_model, use_conv=self.use_conv, name = self.name + "pooler_transform")

    def get_config(self):
        return {
            'n_layers':self.n_layers,
            'num_heads': self.num_heads,
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_segments':self.n_segments,
            'name':self.name,
            'd_model':self.d_model,
            'intermediate_size':self.intermediate_size,
            'activation':self.activation,
            'rate':self.rate,
            'partition_config':self.partition_config
            }

    def build(self, input_shape):
        pass

    def call(self, x, seg, mask, training=False):
        mask = tf.expand_dims(mask, axis=1)
        mask = mask*1e-9

        x = self.embedding(x,seg)
        for layer in self.enc_layers:
            x = layer(x, mask, training=training)
        x = x[:,0]
        x = self.act_out(self.pooler_ffn(x))
        return x

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, intermediate_size, rate=0.1, activation='gelu', partition_config=None, name=None):
        super(BertEncoder, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG

        self.partition_config=partition_config
        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.rate = rate
        self.use_conv = partition_config["use_conv"]

        self.mha = BERTMultiHeadedAttention(num_heads, d_model, activation=activation, partition_config=partition_config, name = "mha")

        self.layernorm1 = DynamicLayerNormalization(epsilon=1e-12, name="attention_layer_norm")
        self.layernorm2 = DynamicLayerNormalization(epsilon=1e-12, name="output_layer_norm")

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.activation = activation

        self.dff = ConfigurableDense(self.intermediate_size, inp_size=self.d_model, use_conv=partition_config["use_conv"], activation=activation, name="intermediate")
        self.out_ffn = ConfigurableDense(self.d_model, inp_size=self.intermediate_size, use_conv=partition_config["use_conv"], name="out")


        self.intermediate_partitions = 1
        self.use_partitions=True

        self.activation1 = activations.get(activation)
        if activation == 'gelu':
            self.activation1 = approx_gelu


    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'intermediate_size':self.intermediate_size,
            'rate':self.rate,
            'partition_config':self.partition_config,
            'name':self.name,
            'activation':self.activation,
            }

    def build(self, input_shape):
        pass

    def call(self, x, mask, training=False):
        attn_output = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output1 = self.dff(out1)
        ffn_output2 = self.out_ffn(ffn_output1)
        ffn_output3 = self.dropout2(ffn_output2, training=training)
        out2 = self.layernorm2(out1 + ffn_output3)
        return out2

class DynamicLayerNormalization(tf.keras.layers.Layer):
    """
    Tensorflow implmentation of integer-only layer norm
    https://github.com/kssteven418/I-BERT
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/ibert/quant_modules.py

    """
    
    def __init__(self, epsilon=1e-6, name=None):
        super(DynamicLayerNormalization, self).__init__(name=name)
        self.epsilon = epsilon

    def get_config(self):
        return {
            'epsilon': self.epsilon,
            'name':self.name
            }

    def build(self, input_shape):
        #gamma
        input_shape = tf.TensorShape(input_shape)
        self.weight = self.add_weight("gamma",shape=[input_shape[-1]],
                initializer='random_normal',
                trainable=True)

        #beta
        self.bias = self.add_weight("beta",shape=[input_shape[-1]],
                initializer='random_normal',
                trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, -1, keepdims=True)
        y = x - mean
        x = y*tf.math.rsqrt(self.epsilon + var)
        x = x * self.weight + self.bias
        return x