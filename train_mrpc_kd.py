import os
import tempfile

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import TransformerModel
import ConvertModel

import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="2"
parser = argparse.ArgumentParser(description='Train MRPC for BERT')
parser.add_argument('student_dir',
                    help='Directory containing student BERT cfg file', type=str)
parser.add_argument('teacher_dir',
                    help='Directory containing teacher BERT cfg file', type=str)
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--task-balance', "--balance", default=0.8, type=float,
                    metavar='balance', help='task balance')
parser.add_argument('--temp', '--temperature', default=1, type=int,
                    metavar='temp', help='temperature value')
parser.add_argument('--sl', '--seq-len', default=128, type=int,
                    metavar='sl', help='sequence length')
parser.add_argument('--warmup-steps', default=30, type=int,
                    metavar='warmup', help='number of warmup steps')

args = parser.parse_args()

student_dir = args.student_dir
teacher_dir = args.teacher_dir

student_name = os.path.basename(student_dir)
teacher_name = os.path.basename(teacher_dir)


epochs = args.epochs
max_seq_length = args.sl
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = args.batch_size
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def fetchRawModel(path, name="transformer"):
    bert_encoder = ConvertModel.from_tf1_checkpoint(path, seq_len=max_seq_length, name=name)
    model = ConvertModel.BERT_Classifier(bert_encoder, 2)
    return model

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

print("Student")
student_model = fetchRawModel(student_dir, name="student")
student_model.summary()

print("Teacher")
teacher_model = fetchRawModel(teacher_dir, name="teacher")
teacher_model.summary()

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=batch_size)

tockenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(student_dir, "vocab.txt"),
    lower_case=True)

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tockenizer.get_special_tokens_dict())

bert_inputs_processor = BertInputProcessor(tockenizer, packer)
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
teacher_metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
teacher_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

teacher_train_loss = tf.keras.metrics.Mean(name='train_loss')
teacher_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='teacher_train_accuracy')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

teacher_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='teacher_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

teacher_test_loss = tf.keras.metrics.Mean(name='teacher_test_loss')
teacher_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='teacher_test_accuracy')

student_model.compile(
    optimizer=optimizer,
    loss=loss_object,
    metrics=metrics)

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
teacher_optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate = warmup_schedule)

teacher_model.compile(
    optimizer=teacher_optimizer,
    loss=teacher_loss_object,
    metrics=teacher_metrics)

best_loss = 999
best_ds = []
for epoch in range(epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    teacher_train_loss.reset_states()
    teacher_train_accuracy.reset_states()
    teacher_test_loss.reset_states()
    teacher_test_accuracy.reset_states()
    transfer_ds = []
    for x_train, y_train_hard in glue_train:
        with tf.GradientTape() as tape:
            y_train_soft = teacher_model(x_train)
            teacher_loss = teacher_loss_object(y_train_hard, y_train_soft)
            transfer_ds.append([x_train, y_train_hard, y_train_soft])
        teacher_gradients = tape.gradient(teacher_loss, teacher_model.trainable_variables)
        teacher_optimizer.apply_gradients(zip(teacher_gradients, teacher_model.trainable_variables))

        teacher_train_loss(teacher_loss)
        teacher_train_accuracy(y_train_hard, y_train_soft)

    for x_test, y_test in glue_validation:
        predictions = teacher_model(x_test)
        t_loss = loss_object(y_test, predictions)

        teacher_test_loss(t_loss)
        teacher_test_accuracy(y_test, predictions)
    
    template = "Teacher: Epoch {}, Train Loss: {:0.3f}, Train Accuracy: {:0.2f}, Val Loss: {:0.3f}, Val Accuracy: {:0.2f}"
    print(template.format(epoch + 1,
                teacher_train_loss.result(),
                teacher_train_accuracy.result() * 100,
                teacher_test_loss.result(),
                teacher_test_accuracy.result() * 100))
    if teacher_test_loss.result() < best_loss:
        best_loss = teacher_test_loss.result()
        best_ds = transfer_ds

    for x_train, y_train_hard, y_train_soft in best_ds:
        with tf.GradientTape() as tape:
            predictions = student_model(x_train)

            hard_target_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_train_hard, tf.int32), logits=predictions)
            soft_target_xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_train_soft, logits=predictions)

            soft_target_xent *= args.temp**2
            total_loss = args.task_balance*hard_target_xent
            total_loss += (1-args.task_balance)*soft_target_xent
        
        student_gradients = tape.gradient(total_loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(student_gradients, student_model.trainable_variables))

        train_loss(total_loss)
        train_accuracy(y_train_hard, predictions)

    for x_test, y_test in glue_validation:
        predictions = student_model(x_test)
        t_loss = loss_object(y_test, predictions)

        test_loss(t_loss)
        test_accuracy(y_test, predictions)


    template = "Student: Epoch {}, Train Loss: {:0.3f}, Train Accuracy: {:0.2f}, Val Loss: {:0.3f}, Val Accuracy: {:0.2f}"
    print(template.format(epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100))

student_model.evaluate(glue_validation)
student_model.save("student-" + student_name + ".h5", include_optimizer=False)
teacher_model.evaluate(glue_validation)
teacher_model.save("teacher-" + teacher_name + ".h5", include_optimizer=False)
#"""