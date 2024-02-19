from utils import *

inference_model = music_inference_model(LSTM_cell, densor, Ty = 50)
inference_summary = summary(inference_model) 