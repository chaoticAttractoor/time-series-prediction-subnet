# developer: taoshi-mbrown
# Copyright © 2023 Taoshi Inc
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Dropout, Layer, LSTM
from keras.mixed_precision import Policy
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import numpy as np
from numpy import ndarray
import resource
import tensorflow as tf
import os 
import torch 
import pandas as pd 
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from lag_llama.gluon.estimator import LagLlamaEstimator

def _get_dataset_options():
    dataset_options = tf.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    return dataset_options


_DATASET_OPTIONS = _get_dataset_options()


class ResourceUsageCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        main_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        main_memory_usage /= 1024
        gpu_memory_usage = tf.config.experimental.get_memory_info("GPU:0")["current"]
        gpu_memory_usage /= 1048576
        print(f" - Memory (MiB): {main_memory_usage:,.2f}, {gpu_memory_usage:,.2f}")




class FoundationalMiningModel:
    def __init__(
        self,
        filename: str,
        mode: str,
        feature_count: int,
        sample_count: int,
        prediction_feature_count: int,
        prediction_count: int,
        prediction_length: int,
        display_memory_usage: bool = False,
        dtype: np.dtype | Policy = Policy("float32"),
    ):
        input_shape = (sample_count, feature_count)
        output_length = prediction_feature_count * prediction_count
        output_shape = (None, output_length)

    
        self._model = None
        self._filename = filename
        self.sample_count = sample_count
        self._prediction_feature_count = prediction_feature_count
        self.prediction_count = prediction_count
        self.prediction_length = prediction_length
        self._display_memory_usage = display_memory_usage
        self.model_dir = filename

            
    def set_model_dir(self, model):
            self.model_dir = model
            return self 
                
    def predict(self, dataset, prediction_length, num_samples=100):
        
        ckpt = torch.load(f"{self.model_dir}/lag-llama.ckpt", map_location=torch.device('cpu')) # Uses GPU since in this Colab we use a GPU.
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
            ckpt_path="lag-llama.ckpt",
            prediction_length=prediction_length,
            context_length=32, # Should not be changed; this is what the released Lag-Llama model was trained with
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],

            batch_size=1,
            num_parallel_samples=100
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=num_samples
        )
        forecasts = list(forecast_it)

        return forecasts
            
                
    
            