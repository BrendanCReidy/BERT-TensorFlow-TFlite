import argparse
import os
import urllib.request
import zipfile

import ConvertModel
import TransformerModel

import tensorflow as tf
import numpy as np

 # Models have different names when downloading
NAME_DICT = {
    "tiny":"uncased_L-2_H-128_A-2",
    "mini":"uncased_L-4_H-256_A-4",
    "medium":"uncased_L-8_H-512_A-8",
    "base":"uncased_L-12_H-768_A-12"
}

URL = "https://storage.googleapis.com/bert_models/2020_02_20"

def get_partition_config(model_type):
    match model_type:
        case "tiny":
            return TransformerModel.EDGETPU_BERT_TINY
        case "mini":
            return TransformerModel.EDGETPU_BERT_MINI
        case "medium":
            return TransformerModel.EDGETPU_BERT_MEDIUM
        case "base":
            return TransformerModel.EDGETPU_BERT_BASE
        case "large":
            return TransformerModel.EDGETPU_BERT_LARGE
        case _:
            raise NotImplementedError(f"No Edge TPU config for model BERT model: {model_type}")

def get_model(model_type):
    model_dir = "models"

    model_full_name = NAME_DICT[model_type]
    model_path = os.path.join(model_dir, f"{model_full_name}")
    if not os.path.exists(os.path.join(model_path, "bert_config.json")):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("Model not found, attempting to download")
        url = f"{URL}/{model_full_name}.zip"
        zip_path = os.path.join(model_path, f"{model_full_name}.zip")
        print("Downloading model...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(model_path)
        print("Cleaning up...")
        os.remove(zip_path)
    return model_path

def export_model(path, model, quant=False, numpy_data=None):

    # Force batch dimension to be 1
    for input_idx in range(len(model.input)):
        model.input[input_idx].set_shape((1,) + model.input[input_idx].shape[1:])
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant:
        # Create dummy data if none provided
        if numpy_data is None:
            numpy_data = [[np.random.random(model.input[i].shape) for i in range(len(model.input))]]

        def representative_dataset():
            for data in numpy_data:
                yield [tf.dtypes.cast(data[i], tf.float32) for i in range(len(data))]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    
if __name__ == "__main__":
    if not os.path.exists("tflite"):
        os.makedirs("tflite")

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str,
                        choices=["float32", "int8", "edgetpu"], default="edgetpu")
    parser.add_argument("--model", type=str,
                        help="Bert model type (tiny, mini, medium, base, large)",
                        choices=["tiny", "mini", "medium", "base", "large"], default="tiny")
    
    parser.add_argument("--seq_len", type=int, help="number of tokens model can process", default=128)
    
    args = parser.parse_args()
    tflite_path = os.path.join("tflite", f"BERT-{args.model}-{args.format}.tflite")

    model_path = get_model(args.model)
    partition_config = None if not args.format=="edgetpu" else get_partition_config(args.model)
    model = ConvertModel.from_tf1_checkpoint(model_path, seq_len=args.seq_len, partition_config=partition_config)
    export_model(tflite_path, model, quant=not (args.format=="float32"))
    print(f"Saved tflite model to: {tflite_path}")