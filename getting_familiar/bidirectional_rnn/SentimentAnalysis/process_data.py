import pandas as pd
import re
import numpy as np
import pickle
from data_preprocessing import retrieve_contents






def generate_vec_and_labels():
    #read tweets
    tweets,labels=read_tweets()
    #mark tweets
    marked_labels=mark_labels(labels)
    #finding word to index dictionary
    word_to_index_dict=word_to_index(tweets)
    #finding index to word dictionary
    index_to_word_dict=index_to_word(word_to_index_dict)
    #list of vector and labels(a list of lists)
    vec_and_labels=create_vec_and_labels(tweets,word_to_index_dict,marked_labels)
    #print(index_to_word_dict)
    vec_and_labels,seq_lengths=length_and_zero_pad(vec_and_labels,45)

    train_vec_and_labels=vec_and_labels[0:10000]
    train_seq_length=seq_lengths[0:10000]

    dev_vec_and_labels=vec_and_labels[10000:12000]
    dev_seq_length=seq_lengths[10000:12000]

    test_vec_and_labels=vec_and_labels[12000:]
    test_seq_length=seq_lengths[12000:]

    save_pickle("data/train.p",train_vec_and_labels,train_seq_length)
    save_pickle("data/test.p",test_vec_and_labels,test_seq_length)
    save_pickle("data/dev.p",dev_vec_and_labels,dev_seq_length)







    with open("data/dataset.p",'wb') as file:
        pickle.dump([vec_and_labels,seq_lengths],file)

    """""
    with open("data/dataset.p",'rb') as file:
        vec_retrieved,seq_retrieved=pickle.load(file)
    """""









def read_tweets():
    tweets=pd.read_csv("data/sentiment-tweets.csv")
    return tweets['text'],tweets["airline_sentiment"]


def split_into_words(sentence):
    #expression for splitting
    _WORD_SPLIT = re.compile("([.,!?\"':;)(])")
    #to store words
    words=[]
    #first split by space
    for space_separated_fragment in sentence.strip().split():
        #and then split by expression
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return words

#split_into_words("@VAmerica What's your 'unmiserable' Problem?Buddys?")
def word_to_index(tweets):
    #empty dict
    word_to_index_dict={}
    #index
    index=1
    #looping over every tweet
    for tweet in tweets:
        #splitting into words
        words=split_into_words(tweet)
        #looping over words
        for word in words:
            #if it exisrt in dict then pass
            if word in word_to_index_dict.keys():
                continue
            else:
                #save the index of word as current index
                word_to_index_dict[word]=index
                #increase the index
                index=index+1

    return word_to_index_dict

def index_to_word(word_to_index_dict):
    #interchange keys and values
    index_to_word_dict={y:x for x,y in word_to_index_dict.items()}
    return index_to_word_dict

def create_vec_and_labels(tweets,word_to_index_dict,labels):
    #for keeping track of labels
    index=0
    #list of lists of vector and labels of form [[[vec1],label1],[[vec2],label2].......]
    vec_and_labels=[]
    #looping over tweets
    for tweet in tweets:
        #list of form [[vec1],label1]
        individual_vec_and_labels = []
        #list of form [vec1]
        vec = []
        #splitiing tweet to words
        words=split_into_words(tweet)
        print(words)
        #for every word in splitted tweet
        for word in words:
            if word is not "":
            #appending index of each word in tweet to vec
                vec.append(word_to_index_dict[word])
        #appending scentence vector to individual_vec_and_labels
        individual_vec_and_labels.append(vec)
        #appending label
        individual_vec_and_labels.append(labels[index])
        #increasing index
        index = index + 1

        vec_and_labels.append(individual_vec_and_labels)

    return vec_and_labels

def mark_labels(labels):
    #coverts labels from string to 1,2,3 for negative,neutral and positive respectively
    marked_labels=[]
    for label in labels:
        if label=="negative":
            marked_labels.append(1)
        if label=="positive":
            marked_labels.append(3)
        if label=="neutral":
            marked_labels.append(2)
    return marked_labels

def get_vocab_size(word_to_index_dict):
    return len(word_to_index_dict)

def to_one_hot(vec_and_labels,vocab_size):
    count=0
    number_of_tweets=len(vec_and_labels)


    for sentence in vec_and_labels:
        print("processed "+str(count)+ " tweets out of "+str(number_of_tweets))

        for i in range(len(sentence[0])):
            one_hot = np.zeros(vocab_size)
            one_hot[sentence[0][i]]=1
            sentence[0][i]=one_hot
        count=count+1
    return vec_and_labels


def verify_one_hot(one_hot,index_to_word_dict):
    print(one_hot[0])

    #print(np.zeros(12))
    for sentence,label in one_hot:

        recons=[]
        for word in sentence:
            index=np.argmax(word)
            recons.append(index_to_word_dict[index])
        print(" ".join(recons))


def length_and_zero_pad(vec_and_labels,max_length):
    seq_lengths=[]
    for vec_and_label in vec_and_labels:
        seq_length=len(vec_and_label[0])
        seq_lengths.append(seq_length)
        for i in range(max_length-seq_length):
            vec_and_label[0].append(0)
    return vec_and_labels,seq_lengths

def save_pickle(filename,vec_and_labels,seq_lengths):
    with open(filename,'wb') as file:
        pickle.dump([vec_and_labels,seq_lengths],file)
generate_vec_and_labels()




