
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

def next_batch(batch_number,batch_size,task):
    if task=="train":
        NUMBER_OF_SENTENCES=7000
        vectors = retrieve_contents("dataset/padded_train_vectors.p")
        seq_lengths=retrieve_contents("dataset/train_seq_length.p")
        labels=retrieve_contents("dataset/one_hot_labels.p")
    if task=="dev":
        NUMBER_OF_SENTENCES=1000

        vectors = retrieve_contents("dataset/dev_vectors.p")
        labels = retrieve_contents("dataset/dev_labels.p")
        seq_lengths = retrieve_contents("dataset/dev_sequence_length.p")


    if NUMBER_OF_SENTENCES-(batch_number*batch_size) <batch_size:
        batch_number=0

    sentence_batch=vectors[batch_number*batch_size:batch_number*batch_size+batch_size]

    seq_lengths_batch=seq_lengths[batch_number*batch_size:batch_number*batch_size+batch_size]
    label_batch=labels[batch_number*batch_size:batch_number*batch_size+batch_size]

    batch_number=batch_number+1
    return sentence_batch,label_batch,seq_lengths_batch,batch_number

def one_hot_labels():
    labels=retrieve_contents("dataset/train_labels.p")

    result = []
    for label in labels:
        one_hot = np.zeros(19)
        one_hot[label - 1] = 1
        result.append(one_hot)
    pickle_contents(result,"dataset/one_hot_labels.p")
    return result


def index_to_word(word_to_index_dict):
    #interchange keys and values
    index_to_word_dict={y:x for x,y in word_to_index_dict.items()}
    return index_to_word_dict



def create_dev_set():
    train_vectors=retrieve_contents("dataset/padded_train_vectors.p")
    train_labels=retrieve_contents("dataset/one_hot_labels.p")
    seq_length=retrieve_contents("dataset/seq_length.p")
    dev_seq_length=seq_length[7000:]
    dev_vectors=train_vectors[7000:]
    dev_labels=train_labels[7000:]
    train_vectors=train_vectors[0:7000]
    train_labels=train_labels[0:7000]
    train_seq_length=seq_length[0:7000]

    print(dev_vectors)
    pickle_contents(train_vectors, "dataset/padded_train_vectors.p")
    pickle_contents(train_labels, "dataset/one_hot_labels.p")
    pickle_contents(dev_vectors,"dataset/dev_vectors.p")
    pickle_contents(dev_labels,"dataset/dev_labels.p")
    pickle_contents(dev_seq_length, "dataset/dev_sequence_length.p")
    pickle_contents(train_seq_length, "dataset/train_seq_length.p")
#create_dev_set()

def dev_set():
    dev_vectors=retrieve_contents("dataset/dev_vectors.p")
    dev_labels=retrieve_contents("dataset/dev_labels.p")
    dev_sequence_length=retrieve_contents("dataset/dev_sequence_length.p")
    print(dev_vectors)

    return dev_vectors,dev_labels,dev_sequence_length

#one_hot_labels()