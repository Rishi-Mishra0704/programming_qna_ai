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

# +
current_dir = os.getcwd()

# Load the data
data_dir = os.path.join(current_dir, "..", 'data')
program_data = pd.read_csv(os.path.join(data_dir, 'program_qna.csv'), sep=',', quotechar='"',)
greetings_data = pd.read_csv(os.path.join(data_dir, 'greetings.csv'))
# -

program_data.head()
program_data.describe()

greetings_data.head()

# ### Check for Duplicates

program_data = program_data.drop_duplicates()
program_data.describe()
