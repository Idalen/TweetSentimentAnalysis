# Please use the following as your fine-tuned model:
# https://drive.google.com/file/d/1vco6ANFYP5UkvkakqoySC71lTb8i3Oe5/view

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from pprint import pprint
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# Folder path containing the fine-tuned model files, link suggested above to download from google drive
model_path = './BERT'

# Set model, tokenizer and prepare classifier
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True) #, top_k=1 )

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Reading csv from file with index already done
df_train = pd.read_csv("../data/tweets_stock_clean.csv").set_index('tweet_id')
df_test = pd.read_csv("../data/tweets_stocks-full_agreement.csv").set_index('tweet_id')

# Extra info as to what is useful and what is useless
targets = ['TRU', 'DIS', 'JOY', 'SAD', 'ANT', 'SUR', 'ANG', 'FEA', 'NEUTRAL']
to_delete = ['conf_tru_dis', 'conf_joy_sad', 'conf_ant_sur',
       'conf_ang_fea', 'num_annot']

# Drop useless columns
df_train.drop(columns=to_delete, inplace=True)
df_test.drop(columns=to_delete, inplace=True)

# Apply our clean_tweet function to all text
df_train['text'] = df_train['text'].apply(lambda text: clean_tweet(text))
df_test['text'] = df_test['text'].apply(lambda text: clean_tweet(text))

inputs = []

print("Collecting tweets and adding into inputs...")
for tweet in tqdm(df_test['text']):
    inputs.append(tweet)

output = classifier(inputs) # This is harder to read = list[list][dict, dict]

predictions = []

print("Transferring output into a predictions list...")
for prediction in tqdm(output): # Make it easier to read = list[dict]['str', 'float']
	predictions.append(list(x for x in prediction))