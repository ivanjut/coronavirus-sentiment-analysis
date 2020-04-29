# Converts csv file of tweet IDs to a line separated txt file to pass into Hydrator
import csv
import pandas as pd

print("Extracting tweet IDs...")
tweet_ids = []
for i in range(1,31):
    if i == 12:
        continue
    
    csv_filename = 'corona_tweets_{}.csv'.format(i)
    tweets = pd.read_csv(csv_filename, header=0, names=['tweet_id', 'sentiment'], dtype={'tweet_id': str, 'sentiment': str})
    if 'tweet_id' not in tweets.columns:
        tweets.columns = ['tweet_id', 'sentiment']
    try:
        sampled = tweets.sample(n=400000)
    except:
        sampled = tweets.sample(frac=1)
    print(sampled.head(2))
    print("SIZE: ", len(sampled))
    for t, row in sampled.iterrows():
        tweet_ids.append(str(row['tweet_id']) + '\n')

    print("Processed csv file {}".format(i))

print("Writing to txt file...")
txt_filename = 'corona_tweets_final.txt'
with open(txt_filename, 'w') as f_write:
    f_write.writelines(tweet_ids)

print("Complete.")

    