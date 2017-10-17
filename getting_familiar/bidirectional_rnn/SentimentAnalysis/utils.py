import numpy as np
import pickle










def next_batch(batch_number,batch_size,task):
    #if not enough examples left then start from begininng
    if task=="train":
        NUMBER_OF_TWEETS=10000
        tweets_and_labels, seq_lengths = pickle.load(open("data/train.p", 'rb'))
    if task=="test":
        NUMBER_OF_TWEETS=2640
        tweets_and_labels, seq_lengths = pickle.load(open("data/test.p", 'rb'))
    if task=="dev":
        NUMBER_OF_TWEETS=2000
        tweets_and_labels, seq_lengths = pickle.load(open("data/dev.p", 'rb'))

    if NUMBER_OF_TWEETS-(batch_number*batch_size) <batch_size:
        batch_number=0
    #creating batches of tweet and label
    tweet_and_label_batch=tweets_and_labels[batch_number*batch_size:batch_number*batch_size+batch_size]
    #seperating tweet and label batches
    tweet_batch,label_batch=seperate_label_and_tweets(tweet_and_label_batch)
    #batch_increment
    seq_lengths_batch=seq_lengths[batch_number*batch_size:batch_number*batch_size+batch_size]
    batch_number=batch_number+1
    return tweet_batch,label_batch,seq_lengths_batch,batch_number




def seperate_label_and_tweets(tweet_and_label_batch):
    #tweet batch
    tweet_batch=[]
    #label batch
    label_batch=[]
    for tweet_and_label in tweet_and_label_batch:
        tweet_batch.append(tweet_and_label[0])
        label_batch.append(tweet_and_label[1])
    return tweet_batch,label_batch


def label_to_one_hot(label_batch):
    result=[]
    for label in label_batch:

        one_hot=np.zeros(3)
        one_hot[label-1]=1
        result.append(one_hot)

    return result


"""""
for i in range(1):
    tweet_batch,label_batch,seq_lengths_batch,batch_number=next_batch(batch_number,20)
    print(tweet_batch)
    print(label_batch)
    print(seq_lengths_batch)
"""""

