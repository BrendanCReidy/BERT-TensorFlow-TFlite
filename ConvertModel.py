#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: Sep 10 2022
# Last Modified: Dec 12 2022
#

import ConvertFromTF1Checkpoint
import os
import tensorflow as tf
import json
import TransformerModel
import re

def BERT_Classifier(backbone_model, classes, use_conv=False):
    backbone = backbone_model
    x = backbone.output
    x = tf.keras.layers.Dropout(0.1)(x)
    x = TransformerModel.ConfigurableDense(classes, inp_size=backbone.output.shape[-1], use_conv=use_conv)(x)
    model = tf.keras.Model(inputs=backbone.input, outputs=x)
    return model

def from_config(configPath, partition_config = None):
    if partition_config==None:
        partition_config=TransformerModel.DEFAULT_PARTITION_CONFIG
    
    with open(configPath) as json_file:
        data = json.load(json_file)
    n_layers = data["num_hidden_layers"]
    num_heads = data["num_attention_heads"]
    vocab_size = data["vocab_size"]
    seq_len = data["max_position_embeddings"]
    n_segments = data["type_vocab_size"]
    d_model = data["hidden_size"]
    intermediate_size = data["intermediate_size"]
    activation = data["hidden_act"]

    x = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_word_ids", ragged=False)
    seg = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_type_ids", ragged=False)
    mask = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_mask", ragged=False)
    custom_encoder = TransformerModel.BERT(n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size,
        activation=activation, partition_config=partition_config, name="transformer")(x, seg, mask)
    encoder_model = tf.keras.Model(inputs=[x, seg, mask], outputs=[custom_encoder])
    return encoder_model

def from_tf1_checkpoint(model_dir, partition_config=None, seq_len=128, name="transformer"):
    if partition_config==None:
        partition_config=TransformerModel.DEFAULT_PARTITION_CONFIG
    return ConvertFromTF1Checkpoint.from_tf1_checkpoint(
        os.path.join(model_dir, "bert_model.ckpt"),
        os.path.join(model_dir, "bert_config.json"),
        partition_config, seq_len=seq_len, name=name
    )
