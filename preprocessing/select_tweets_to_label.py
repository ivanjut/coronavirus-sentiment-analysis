import numpy as np
import pandas as pd
import time


print("Reading full csv...")
overall_start = time.time()
full_tweets = pd.read_csv('filtered_tweets.csv', error_bad_lines=False)
reading_time = time.time() - overall_start
print("Reading time: ", reading_time)
print(len(full_tweets), " tweets")

print("Sampling...")
start = time.time()
sampled = full_tweets.sample(n=2000)
print("Sampling time: ", time.time() - start)
print("Sampled ", len(sampled), " tweets")
print(sampled.head(10)['text'])

print("Writing to new csv...")
sampled.to_csv('sampled_tweets_to_label.csv')
print("Done: ", time.time() - overall_start)
