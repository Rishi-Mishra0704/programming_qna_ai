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


# +

def build_rnn_model(vocab_size, embedding_dim, max_sequence_length):

    model = Sequential()

    # Embedding layer to convert words to vectors
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

    # LSTM layer (with dropout for regularization)
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

    model.add(LSTM(units=128))  # Another LSTM layer

    # Dense layer to output probability for each word in the vocabulary
    model.add(Dense(units=vocab_size, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# -

inputs = preprocessed_program_data.iloc[:, :-2].values  # All columns except the last two
targets = preprocessed_program_data.iloc[:, -2:].values  # Last two columns are the output


# +
X = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))

# Targets are the next word predictions, so it will be a one-hot encoded vector of size vocab_size.
y = targets

# +
# Define parameters
vocab_size = inputs.shape[1]  # The number of unique tokens in the data (columns in the CSV)
embedding_dim = 50  # You can tune this value
max_sequence_length = 1  # Since each row is a sequence of length 1

# Build the RNN model
model = build_rnn_model(vocab_size, embedding_dim, max_sequence_length)

# Summarize the model
model.summary()

# -

print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")


# +
# Create a tokenizer for the labels (words in `y`)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_program_data.columns)  # Assuming these are the vocabulary

# Convert the words in `y` to integer indices
y_indices = tokenizer.texts_to_sequences(y)

# Since y_indices will be a list of lists (one per sample), we need to flatten it into a 1D array
y_indices = np.array([item[0] for item in y_indices])

# Now one-hot encode the labels
y_one_hot = to_categorical(y_indices, num_classes=vocab_size)


# +
# Train the model
history = model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2, verbose=2)

# Optionally: Save the model
model.save(os.path.join("..","model",'text_generation_rnn_model.h5'))

