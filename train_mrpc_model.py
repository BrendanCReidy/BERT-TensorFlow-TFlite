import os
import tempfile

from datasets import load_dataset
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import numpy as np

import TransformerModel
import ConvertModel

import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser(description='Train MRPC for BERT')
parser.add_argument('model_dir',
                    help='Directory containing BERT cfg file', type=str)
parser.add_argument('--epochs', default=4, type=int, metavar='N',
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
model_name = os.path.basename(model_dir)

epochs = args.epochs
max_seq_length = args.sl
batch_size = args.batch_size

def fetchRawModel():
    bert_encoder = ConvertModel.from_tf1_checkpoint(model_dir, seq_len=max_seq_length)
    bert_classifier = ConvertModel.BERT_Classifier(bert_encoder, 2)
    return bert_classifier

bert_classifier = fetchRawModel()
bert_classifier.summary()

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=batch_size)

torch_dataset = load_dataset("glue", "mrpc")

print("Remapping test set to conform with tensorflow")
glue_test = {"sentence1":[], "sentence2":[], "label":[]}
for data in torch_dataset['test']:
    glue_test["sentence1"].append(data["sentence1"])
    glue_test["sentence2"].append(data["sentence2"])
    glue_test["label"].append(data["label"])

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
glue_test, glue_labels = bert_inputs_processor(glue_test)

train_data_size = info.splits['train'].num_examples
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = args.warmup_steps
initial_learning_rate=args.lr

cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, steps_per_epoch*epochs)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = cosine_lr,
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

glue_test = glue_test
glue_labels = glue_labels

#bert_classifier.evaluate(glue_test, glue_labels)
predictions = bert_classifier(glue_test)
max_out = np.argmax(predictions,axis=1)
actual = np.array(glue_labels)
acc = np.mean(actual==max_out)
print("Test accuracy:", acc)
def f1(actual, predicted, label):

    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def f1_macro(actual, predicted):
    # `macro` f1- unweighted mean of f1 per label
    return np.mean([f1(actual, predicted, label) 
        for label in np.unique(actual)])

f1_val = f1_macro(actual, max_out)
print("Test F1:", f1_val)


bert_classifier.save(model_name + ".h5", include_optimizer=False)
#"""