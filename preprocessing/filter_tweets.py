import numpy as np
import pandas as pd
import time


print("Reading full csv...")
overall_start = time.time()
full_tweets = pd.read_csv('corona_tweets_data.csv', error_bad_lines=False)
reading_time = time.time() - overall_start
print("Reading time: ", reading_time)
print(len(full_tweets), " tweets")

print("Filtering...")
start = time.time()
filtered = full_tweets[(~full_tweets['text'].str.startswith('RT')) & (~full_tweets['text'].str.contains('@')) & (~full_tweets['text'].str.contains('http'))]
print("Filtering time: ", time.time() - start)
print("Filtered ", len(filtered), " tweets")
print(filtered.head(10)['text'])

print("Writing to new csv...")
filtered.to_csv('filtered_tweets.csv')
print("Done: ", time.time() - overall_start)
