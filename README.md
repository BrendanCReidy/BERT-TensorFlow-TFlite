# BERT->TensorFlow2->TFlite

This repository contians a pipeline for:
1.  Loading TF1 BERT models in TF2
2.  Training BERT models for downstream tasks (with or without knowledge distillation)
3.  Exporting BERT models as TFLite files

### Getting started

This codebase was designed to load models from [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert). 

Example: BERT-Tiny
```
mkdir models && cd models
mkdir uncased_L-2_H-128_A-2 && cd uncased_L-2_H-128_A-2
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip
unzip uncased_L-2_H-128_A-2.zip
rm uncased_L-2_H-128_A-2.zip
```
BERT-Base
```
mkdir uncased_L-12_H-768_A-12 && cd uncased_L-12_H-768_A-12
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
rm uncased_L-12_H-768_A-12.zip
```

### Training the model

Without knowledge distilation:
```
python3 train_mrpc_model.py model_dir
```
For BERT-tiny:
```
python3 train_mrpc_model.py models/uncased_L-2_H-128_A-2
```

With knowledge distilation:
```
python3 train_mrpc_kd.py student_model_dir teacher_model_dir
```
For BERT-Tiny student and BERT-Base teacher:
```
python3 train_mrpc_kd.py models/uncased_L-2_H-128_A-2 models/uncased_L-12_H-768_A-12
```
