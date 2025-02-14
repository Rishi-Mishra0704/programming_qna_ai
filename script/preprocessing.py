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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

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

program_data = program_data.drop_duplicates(subset=['input'], keep='first')
program_data = program_data.drop_duplicates(subset=['output'], keep='first')
program_data.describe()

program_data["input_tokens"] = program_data["input"].apply(lambda x: [token.text.lower() for token in nlp(x)])
program_data["output_tokens"] = program_data["output"].apply(lambda x: [token.text.lower() for token in nlp(x)])


program_data.head()

greetings_data["input_tokens"] = greetings_data["input"].apply(lambda x: [token.text.lower() for token in nlp(x)])
greetings_data["output_tokens"] = greetings_data["output"].apply(lambda x: [token.text.lower() for token in nlp(x)])
greetings_data.head()


def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


program_data["input_lemmas"] = program_data["input_tokens"].apply(lemmatize)
program_data["output_lemmas"] = program_data["output_tokens"].apply(lemmatize)
program_data.head()

tf_idf = TfidfVectorizer()

program_data['input_text'] = program_data['input_tokens'].apply(lambda x: ' '.join(x))
program_data['output_text'] = program_data['output_tokens'].apply(lambda x: ' '.join(x))


program_x_input = tf_idf.fit_transform(program_data['input_text'])
program_x_output = tf_idf.fit_transform(program_data['output_text'])

program_data.shape

greetings_data["input_lemmas"] = greetings_data["input_tokens"].apply(lemmatize)
greetings_data["output_lemmas"] = greetings_data["output_tokens"].apply(lemmatize)

greetings_data['input_text'] = greetings_data['input_tokens'].apply(lambda x: ' '.join(x))
greetings_data['output_text'] = greetings_data['output_tokens'].apply(lambda x: ' '.join(x))

greeting_x_input = tf_idf.fit_transform(greetings_data['input_text'])
greeting_x_output = tf_idf.fit_transform(greetings_data['output_text'])

greetings_data.shape

# +
# Apply TF-IDF transformation to the input text
program_tfidf = tf_idf.transform(program_data['input_text'])

# Convert the TF-IDF result to a DataFrame
program_tfidf_df = pd.DataFrame(program_tfidf.toarray(), columns=tf_idf.get_feature_names_out())

# Add input and output columns back to the DataFrame
program_tfidf_df['input'] = program_data['input']
program_tfidf_df['output'] = program_data['output']

# Save the DataFrame to a CSV file
program_tfidf_df.to_csv(os.path.join(data_dir, 'program_preprocessed_data.csv'), index=False)


# +
greeting_tfidf = tf_idf.transform(greetings_data['input_text'])

# Convert the TF-IDF result to a DataFrame
greeting_tfidf = pd.DataFrame(program_tfidf.toarray(), columns=tf_idf.get_feature_names_out())

# Add input and output columns back to the DataFrame
greeting_tfidf['input'] = program_data['input']
greeting_tfidf['output'] = program_data['output']

# Save the DataFrame to a CSV file
greeting_tfidf.to_csv(os.path.join(data_dir, 'greeting_preprocessed_data.csv'), index=False)
