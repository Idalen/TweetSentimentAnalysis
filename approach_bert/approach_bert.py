# Please use the following as your fine-tuned model:
# https://drive.google.com/file/d/1vco6ANFYP5UkvkakqoySC71lTb8i3Oe5/view

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from pprint import pprint
import pandas as pd

#Folder path containing the fine-tuned model files
model_path = './BERT'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

eval_data = pd.read_csv('../data/tweets_stocks-full_agreement.csv')

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
