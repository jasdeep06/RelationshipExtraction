
import pickle
import re
import numpy as np

def retrieve_contents(filename):
    content=pickle.load(open(filename,"rb"))
    return content

def pickle_contents(content,filename):
    pickle.dump(content,open(filename,'wb'))

def split_into_words(sentence):
    #expression for splitting
    _WORD_SPLIT = re.compile("([.,!?\"':;)<>(])")
    #to store words
    words=[]
    #first split by space
    for space_separated_fragment in sentence.strip().split():
        #and then split by expression
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return words



def word_to_index():
    word_to_index_dict={}
    index=1
    sentences=retrieve_contents("dataset/train_sentences.p")
    for sentence in sentences:
        sentence=sentence[1:-1]
        words=split_into_words(sentence)
        for word in words:
            if word!="":
                if word in word_to_index_dict.keys():
                    continue
                else:
                    word_to_index_dict[word]=index
                    index=index+1
    return word_to_index_dict



def convert_sentences_to_vectors():
    word_to_index_dict=word_to_index()
    print(word_to_index_dict)
    vectors=[]
    sentences=retrieve_contents("dataset/train_sentences.p")
    for sentence in sentences:
        vector=[]
        sentence = sentence[1:-1]
        words = split_into_words(sentence)
        for word in words:
            if word !="":
                vector.append(word_to_index_dict[word])
        vectors.append(vector)
    pickle_contents(vectors,"dataset/train_vectors.p")

    return vectors

def determine_length():
    vecs=retrieve_contents("dataset/train_vectors.p")
    length=[]
    for vec in vecs:
        length.append(len(vec))
    print(length)
    pickle_contents(length, "dataset/seq_length.p")

def next_batch(batch_number,batch_size):

    NUMBER_OF_SENTENCES=8000
    vectors = retrieve_contents("dataset/train_vectors.p")
    seq_lengths=retrieve_contents("dataset/seq_length.p")
    labels=retrieve_contents("dataset/train_labels.p")


    if NUMBER_OF_SENTENCES-(batch_number*batch_size) <batch_size:
        batch_number=0

    sentence_batch=vectors[batch_number*batch_size:batch_number*batch_size+batch_size]

    seq_lengths_batch=seq_lengths[batch_number*batch_size:batch_number*batch_size+batch_size]
    label_batch=labels[batch_number*batch_size:batch_number*batch_size+batch_size]

    batch_number=batch_number+1
    return sentence_batch,label_batch,seq_lengths_batch,batch_number

determine_length()





