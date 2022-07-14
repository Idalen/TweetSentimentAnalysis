# Please use the following as your fine-tuned model:
# https://drive.google.com/file/d/1vco6ANFYP5UkvkakqoySC71lTb8i3Oe5/view

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from pprint import pprint
import pandas as pd
import numpy as np
import re

#Folder path containing the fine-tuned model files
model_path = './BERT'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

df_train = pd.read_csv("../data/tweets_stock_clean.csv").set_index('tweet_id')
df_test = pd.read_csv("../data/tweets_stocks-full_agreement.csv").set_index('tweet_id')

targets = ['TRU', 'DIS', 'JOY', 'SAD', 'ANT', 'SUR', 'ANG', 'FEA']

to_delete = ['NEUTRAL', 'conf_tru_dis', 'conf_joy_sad', 'conf_ant_sur',
       'conf_ang_fea', 'num_annot']

df_train.drop(columns=to_delete, inplace=True)
df_test.drop(columns=to_delete, inplace=True)

df_train['text'] = df_train['text'].apply(lambda text: clean_tweet(text))
df_test['text'] = df_test['text'].apply(lambda text: clean_tweet(text))

inputs = [
	'Eu te amo',
	'Eu acho que você é uma ótima pessoa',
	'Eu odeio aquele cara',
	]

output = classifier(inputs)

predictions = []

for prediction in output:
	predictions.append(list(x for x in prediction))

pprint(predictions)
