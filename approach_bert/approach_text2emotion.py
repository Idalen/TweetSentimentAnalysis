# This is the file for approach 1 for tweet sentiment analysis
import pandas as pd
import text2emotion as te

# Read the csv file
training_data = pd.read_csv('../tweets_stock_clean.csv')
eval_data = pd.read_csv('../tweets_stocks-full_agreement.csv')

print(eval_data)

print(te.get_emotion("Have a very good day, son!"))