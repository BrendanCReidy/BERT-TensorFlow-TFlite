import os
import tempfile

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import TransformerModel
import ConvertModel

import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='Train MRPC for BERT')
parser.add_argument('model_dir',
                    help='Directory containing BERT cfg file', type=str)
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--sl', '--seq-len', default=128, type=int,
                    metavar='sl', help='sequence length')
parser.add_argument('--warmup-steps', default=30, type=int,
                    metavar='warmup', help='number of warmup steps')

args = parser.parse_args()

model_dir = args.model_dir
model_name = model_dir
if "/" in model_dir:
    model_name = model_dir.split("/")[-1]

epochs = args.epochs
max_seq_length = args.sl
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = args.batch_size
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def fetchRawModel(batch_size=None):
    bert_encoder = ConvertModel.from_tf1_checkpoint(model_dir, seq_len=max_seq_length)
    bert_classifier = ConvertModel.BERT_Classifier(bert_encoder, 2)
    return bert_classifier

bert_classifier = fetchRawModel()
bert_classifier.summary()

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=batch_size)

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(model_dir, "vocab.txt"),
    lower_case=True)

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())

class BertInputProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, packer):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer

    def call(self, inputs):
        tok1 = self.tokenizer(inputs['sentence1'])
        tok2 = self.tokenizer(inputs['sentence2'])

        packed = self.packer([tok1, tok2])

        if 'label' in inputs:
            return packed, inputs['label']
        else:
            return packed

bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)

train_data_size = info.splits['train'].num_examples
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = args.warmup_steps
initial_learning_rate=args.lr

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate = warmup_schedule)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.fit(
      glue_train,
      validation_data=(glue_validation),
      batch_size=batch_size,
      epochs=epochs)

bert_classifier.evaluate(glue_validation)
bert_classifier.save(model_name + ".h5", include_optimizer=False)