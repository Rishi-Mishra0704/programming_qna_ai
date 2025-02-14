# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# +
current_dir = os.getcwd()

# Load the data
data_dir = os.path.join(current_dir, "..", 'data')
preprocessed_program_data = pd.read_csv(os.path.join(data_dir, 'program_preprocessed_data.csv'), sep=',', quotechar='"',)
preprocessed_greetings_data = pd.read_csv(os.path.join(data_dir, 'greeting_preprocessed_data.csv'))
