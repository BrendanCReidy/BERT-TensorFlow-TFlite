# BERT &rarr; TensorFlow2 &rarr; TFlite

### Getting Started
Clone
```
git clone https://github.com/BrendanCReidy/BERT-TensorFlow-TFlite.git
cd BERT-TensorFlow-TFlite
```

(Optional) Create Virtual Environment (Tested on python 3.10.16)
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```
### Export Model
Usage of export.py
```
usage: export.py [-h] [--format {float32,int8,edgetpu}] [--model {tiny,mini,medium,base}] [--seq_len SEQ_LEN]

options:
  -h, --help            show this help message and exit
  --format {float32,int8,edgetpu}
  --model {tiny,mini,medium,base}
                        Bert model type (tiny, mini, medium, base)
  --seq_len SEQ_LEN     number of tokens model can process
```
Example
```
python3 export.py --format edgetpu --model mini
```
Output
```
Saved tflite model to: tflite/BERT-mini-edgetpu.tflite
```
*Note: You still need to use the EdgeTPU compiler on the TFLite if you want to deploy this on the coral TPU*

### Training (TFLite export not supported yet)

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
