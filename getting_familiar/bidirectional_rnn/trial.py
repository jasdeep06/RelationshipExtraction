
"""""

cell=tf.nn.rnn_cell.LSTMCell(num_units=64,state_is_tuple=False)
outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,dtype=tf.float64,sequence_length=X_lengths,inputs=X)
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

output_f,output_b=outputs
state_f,state_b=states
print(sess.run(output_f))
print(sess.run(states))
"""""

"""""
f=open("data/sample_text.txt")
text=f.read()
#splitted_text=re.split(" |.",text)
splitted_text=re.findall(r"[\w']+|[.,!?;]", text)

max_length=20


def word_to_index(words):
    index=0
    word_to_index_dict={}
    for word in words:
        if word in word_to_index_dict.keys():
            continue
        else:
            word_to_index_dict[word]=index
            index=index+1
    return word_to_index_dict

def index_to_word(word_to_index):
    index_to_word={y:x for x,y in word_to_index.items()}
    return index_to_word

word_to_index=word_to_index(splitted_text)
print(word_to_index["1500s"])

index_to_word=index_to_word(word_to_index)
print(index_to_word)

def number_of_sentences(text):
    number=0
    for char in text:
        if char==".":
            number=number+1
        else:
            continue
    return number


#print(number_of_sentences(text))
def next(current_index,batch_size):
   
    required_punctuation=string.punctuation.replace(".","")
    word_count=0
    word_placeholder=""
    batch_x=[]
    internal_batch=[]
    end_token=0
    sequence_lengths=[]
    #character_count=0

    if current_index == len(text):
        current_index=0

    for char in text[current_index:]:

        print(char)

        current_index=current_index+1

       
        if char not in required_punctuation and char !="." and char !=" ":
            word_placeholder += char

        if char==" ":
            internal_batch.append(word_to_index[word_placeholder])
            word_placeholder=""
            word_count=word_count+1


        if char==".":
            internal_batch.append(word_to_index[word_placeholder])
            word_placeholder = ""
            batch_x.append(internal_batch)
            internal_batch=[]
            word_count=word_count+1
            end_token=end_token+1
        if char in required_punctuation:
            internal_batch.append(word_to_index[word_placeholder])
            internal_batch.append(word_to_index[char])
            word_placeholder=""
            word_count=word_count+1


        if end_token==batch_size:
            return batch_x,current_index




print(next(0,2))



#print(string.punctuation)


#print(next(22,33))
        
"""""
import pickle

content=pickle.load(open("data/dev.p",'rb'))

print(content)