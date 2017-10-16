import numpy as np
import pickle

NUMBER_OF_TWEETS=14641

tweets_and_labels=pickle.load(open("data/dataset.p",'rb'))

batch_number=0

def next_batch(batch_number,batch_size):
    #if not enough examples left then start from begininng
    if NUMBER_OF_TWEETS-(batch_number*batch_size) <batch_size:
        batch_number=0
    #creating batches of tweet and label
    tweet_and_label_batch=tweets_and_labels[batch_number*batch_size:batch_number*batch_size+batch_size]
    #seperating tweet and label batches
    tweet_batch,label_batch=seperate_label_and_tweets(tweet_and_label_batch)
    #batch_increment
    batch_number=batch_number+1
    return tweet_batch,label_batch,batch_number




def seperate_label_and_tweets(tweet_and_label_batch):
    #tweet batch
    tweet_batch=[]
    #label batch
    label_batch=[]
    for tweet_and_label in tweet_and_label_batch:
        tweet_batch.append(tweet_and_label[0])
        label_batch.append(tweet_and_label[1])
    return tweet_batch,label_batch

for i in range(5):
    tweet_batch,label_batch,batch_number=next_batch(batch_number,20)
    print(tweet_batch)
    print(label_batch)