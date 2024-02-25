# developer: taoshi-mbrown
# Copyright © 2023 Taoshi Inc

import numpy as np
from numpy import ndarray
import resource
import os 
from neuralforecast import NeuralForecast
import pandas as pd 
from sklearn import mean_squared_error
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




class AdaptiveModelStack:
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
        
            
    def load_models(self):
            dirs = os.listdir(self.model_dir) # set to full path
            model_dirs = [os.path.join(self.model_dir, entry) for entry in dirs if os.path.isdir(os.path.join(self.model_dir, entry))]

            dict = [ NeuralForecast.load(model) for model in model_dirs ]
            self.loaded_models = dict 
            return self
        
        
    def select_model(self,df,ground_truth,length=25):
            #length = length - 1
            k = 0.1
            weights= np.exp(-k * np.arange(length))
                        
            predictions = [ model.predict(df).drop(columns='ds') for model in self.loaded_models] 
            errors = [ mean_squared_error(prediction.values[0:length],ground_truth.values[0:length] ,sample_weight=weights, squared=False)  for prediction in predictions]
            min_error = errors.index(min(errors)) # find index 
            return self.loaded_models[min_error]
        
            
    def average_model(self,df):
            predictions = [ model.predict(df).drop(columns='ds') for model in self.loaded_models] 
            prediction_av = pd.concat(predictions,axis=1).apply(lambda x: np.average(x),axis=1).values
           #  min_eprerror = errors.index(min(errors)) # find index 
            return prediction_av
        
            
    def weighted_average_model(self,df,ground_truth,true_pred_df,length=25):
            k = 0.1
            weights= np.exp(-k * np.arange(length))
    
            predictions = [ model.predict(df).drop(columns='ds') for model in self.loaded_models] 
            errors = [ mean_squared_error(prediction.values[0:length],ground_truth.values[0:length] ,sample_weight=weights, squared=False)  for prediction in predictions]
            pweights= [1/error for error in errors] 
            
            real_predictions = [ model.predict(true_pred_df).drop(columns='ds') for model in self.loaded_models] 
            prediction_av = pd.concat(real_predictions,axis=1).apply(lambda x: np.average(x, weights=pweights),axis=1).values

            return prediction_av
       