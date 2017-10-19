
import re
import pickle
import os

def seperate_sentences():
    f = open("dataset/TRAIN_FILE.txt")
    sentences=[]
    text = f.read()
    splitted = re.split('\n', text)
    for i in range(0, 32000, 4):
        nest_split = splitted[i].split()
        nest_split = nest_split[1:]
        sentence=" ".join(nest_split)
        sentences.append(sentence)
    print(sentences)

    pickle_contents(sentences,"dataset/train_sentences.p")



def pickle_contents(content,filename):
    pickle.dump(content,open(filename,'wb'))

def retrieve_contents(filename):
    content=pickle.load(open(filename,"rb"))
    return content

def seperate_relations():
    f = open("dataset/TRAIN_FILE.txt")
    relations=[]
    text = f.read()
    splitted = re.split('\n', text)
    for i in range(1, 32000, 4):
        relation=splitted[i]
        relations.append(relation)
    print(relations)


    pickle_contents(relations,"dataset/train_relations.p")

def generate_relation_key():
    relation_key={}
    index=1
    relations=retrieve_contents("dataset/train_relations.p")
    distinct_relations=list(set(relations))
    for relation in distinct_relations:
        relation_key[relation]=index
        index=index+1
    print(relation_key)
    pickle_contents(relation_key,"dataset/relation_key.p")





    #print(relations)

def label_relations():
    relation_key=retrieve_contents("dataset/relation_key.p")
    relations=retrieve_contents("dataset/train_relations.p")
    labels=[]
    for relation in relations:
        labels.append(relation_key[relation])
    pickle_contents(labels,"dataset/train_labels.p")


def pad_vector():
    max_length=112
    vecs=retrieve_contents("dataset/train_vectors.p")
    seq_length=retrieve_contents("dataset/seq_length.p")
    index=0
    for vec in vecs:
        for i in range(max_length - seq_length[index]):
            vec.append(0)
        index=index+1
    pickle_contents(vecs,"dataset/padded_train_vectors.p")
    return vecs
pad_vector()