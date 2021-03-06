# Please use the following as your fine-tuned model:
# https://drive.google.com/file/d/1srCYkS3dMR41BY0Y3WzMpCydJECfonUj/view?usp=sharing

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

threshold = 0.003

# Folder path containing the fine-tuned model files, link suggested above to download from google drive
model_path = './BERT_large'

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
alt_targets = ['otimismo', 'nojo', 'alegria', 'tristeza', 'nervosismo', 'surpresa', 'raiva', 'medo', 'neutro']
to_delete = ['conf_tru_dis', 'conf_joy_sad', 'conf_ant_sur', 'conf_ang_fea', 'num_annot']

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

clean_list = []

print("Adding only the desirable alt_targets to final list...")
for prediction in tqdm(predictions):
    clean_row = []
    for item in prediction:
        if item['label'] in alt_targets:
            clean_row.append(item['score'])
    clean_list.append(clean_row)

df_pred = pd.DataFrame(clean_list, columns=targets) # We now use targets instead of alt_targets because it's more useful to use the english names now
df_true = pd.DataFrame(df_test[targets], columns=targets) # These are the true answers to compare to

# Keeping only values above threshold and using True and False for their results
for target in targets:
    df_pred[target] = df_pred[target] > threshold 
    df_true[target] = df_true[target] > threshold

y_pred = np.array(df_pred).flatten()
y_true = np.array(df_true).flatten()

# Write final results to file with datetime name for safety from overwriting
currentDT = datetime.datetime.now()
currentDTstr = str(currentDT).replace(" ", "_").replace(":", "_").replace(".", "_").replace("-", "_")

df_pred.to_csv("confusao_pred_"+currentDTstr+".csv", index = False, encoding='utf-8')
df_true.to_csv("confusao_true_"+currentDTstr+".csv", index = False, encoding='utf-8')

with open("scores_"+currentDTstr+".txt", "w") as f:
	f.write("Accuracy: "+str(round(100*accuracy_score(y_pred, y_true),2))+' %\n')
	f.write("Precision: "+str(round(100*precision_score(y_pred, y_true),2))+' %\n')
	f.write("Recall: "+str(round(100*recall_score(y_pred, y_true),2))+' %\n')
	f.write("F1-score: "+str(round(100*f1_score(y_pred, y_true), 2))+' %\n')
	f.write("\n")

	for target in targets:
		f.write(target)
		f.write("\n")
		f.write("   Accuracy: "+str(round(100*accuracy_score(df_pred[target], df_true[target]),2))+' %\n')
		f.write("   Precision: "+str(round(100*precision_score(df_pred[target], df_true[target]),2))+' %\n')
		f.write("   Recall: "+str(round(100*recall_score(df_pred[target], df_true[target]),2))+' %\n')
		f.write("   F1-score: "+str(round(100*f1_score(df_pred[target], df_true[target]), 2))+' %\n')
		f.write("\n")