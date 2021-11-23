import re
import pickle
import nltk 
# nltk.download('wordnet')
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import numpy as np
from numpy.core.fromnumeric import shape
import random
# import tensorflow as tf
from keras.models import load_model

import re
import os
# hdf5_store = HDF5Store('hdf5_newdataset.h5','X', shape=(1000,36524))
dataset_file = open(os.path.dirname(os.path.realpath(__file__)) + '/data/dataset_to_use.pkl', "rb")
_EMBMAT_FILE_READ = open(os.path.dirname(os.path.realpath(__file__)) + '/data/emb_matrix.pkl', "rb")
_WORD2ID_FILE_READ = open(os.path.dirname(os.path.realpath(__file__)) + '/data/word2id.pkl', "rb")
_ID2WORD_FILE_READ = open(os.path.dirname(os.path.realpath(__file__)) + '/data/id2word.pkl', "rb")
_VOCAB_FILE_READ = open(os.path.dirname(os.path.realpath(__file__)) + '/data/vocab.pkl', "rb")
emb_matrix = pickle.load(_EMBMAT_FILE_READ)
word2id = pickle.load(_WORD2ID_FILE_READ)
id2word = pickle.load(_ID2WORD_FILE_READ)
token_to_index_array = pickle.load(_VOCAB_FILE_READ)

padded_node = 'O-3'

current_dataset = pickle.load(dataset_file)
_WORD_SPLIT = re.compile(r"([.,!?\"':;()/-])")

# model_path = os.path.dirname(os.path.realpath(__file__)) + '/nn_model'
# print(model_path)
# deepModel = load_model('Django usable model')



def preprocess_instruction(sentence):
    # change "office-12" or "office12" to "office 12"
    # change "12-office" or "12office" to "12 office"
    _WORD_NO_SPACE_NUM_RE = r'([A-Za-z]+)\-?(\d+)'
    _NUM_NO_SPACE_WORD_RE = r'(\d+)\-?([A-Za-z]+)'
    new_str = re.sub(_WORD_NO_SPACE_NUM_RE, lambda m: m.group(1) + ' ' + m.group(2), sentence)
    new_str = re.sub(_NUM_NO_SPACE_WORD_RE, lambda m: m.group(1) + ' ' + m.group(2), new_str)
    
    lemma = WordNetLemmatizer()
    
    # correct common typos.
    correct_error_dic = {'rom': 'room', 'gout': 'go out', 'roo': 'room',
                         'immeidately': 'immediately', 'halway': 'hallway',
                         'office-o': 'office 0', 'hall-o': 'hall 0', 'pas': 'pass',
                         'offic': 'office', 'leftt': 'left', 'iffice': 'office', 'reachthe': 'reach the',
                         'entrace':'entrance'
                         }
    
    for err_w in correct_error_dic:

        find_w = ' ' + err_w + ' '
        replace_w = ' ' + correct_error_dic[err_w] + ' '
        new_str = new_str.replace(find_w, replace_w)

    sen_list = []
    # Lemmatize words
    for word in new_str.split(' '):
        try:
            word = lemma.lemmatize(word)
            if len(word) > 0 and word[-1] == '-':
                word = word[:-1]
            if word:
                sen_list.append(word)
        except UnicodeDecodeError:
            continue
            # print("unicode error ", word, new_str)
    return sen_list


def simple_tokenizer(sentence):
    #removes characters like exclamation mark,comma,dot brackets etc and returns a tokenized array
    words = []
    prepocessed_sen_list = preprocess_instruction(sentence.strip())
    for space_separated_fragment in prepocessed_sen_list:
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]


def convert_nautural_sequence_into_embedding(emb_matrix, word2id, id2word, natural_language_sequence):
    #returns an embedding vector corresponding to a natural language sequence
    '''
    First tokenise using simple_tokenizer  
    returns embedding matrix of the NL instruction
    '''
    NL_tokenized = simple_tokenizer(natural_language_sequence.lower())
    # print(NL_tokenized)
    token_ids = [word2id[token] for token in NL_tokenized]
    padding_length = 265
    # print(emb_matrix.shape)
    nl_emb_matrix = [emb_matrix[idx] for idx in token_ids]
    for _ in range(padding_length-len(nl_emb_matrix)):
        nl_emb_matrix.append(np.zeros(shape=(100,)))
   
    nl_emb_matrix = np.stack(nl_emb_matrix, axis=0)
    # print(nl_emb_matrix)
    # print(nl_emb_matrix.shape)

    
    return nl_emb_matrix # numpy 2D array with shape -> (number of tokens in NL, GloVE dimensions)

def get_one_hot_encoded_sequence(token_to_index_array, array_for_conversion):
    #converts a robot language sequence into an onehot encoded sequence
    '''
    token_to_index_array - RL token2id
    array_for_conversion - subset path only one tokenized array
    Return array of one hot encoded vectors of the subset paths
    '''
    all_tokens = token_to_index_array[0]
    tokens2id = token_to_index_array[1]

    
    # Define a max length -> padding size
    max_length = 46


    one_hot_vector = np.zeros(shape = (max_length+1, len(all_tokens)))
    
    # # One hot encoding main 
    # for i,  subset in enumerate(array_for_conversion):
    for j, token in enumerate(array_for_conversion):
        
        idx = tokens2id[token]
        one_hot_vector[j, idx] = 1


    return one_hot_vector # numpy array of shape (number of subsets in array_for_conversion, max_length, length of all tokens in BM)




def embed_ground_truth(ground_truth):
    behaviors = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl','end']
    behavior2idx = {beh : idx for idx, beh in enumerate(behaviors)}
    
    one_hot_vector = np.zeros(shape = (len(behaviors)))

    one_hot_vector[behavior2idx[ground_truth]] = 1

    return one_hot_vector


def embed_ground_truth_labelencoded(ground_truth):
    behaviors = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl','end']
    behavior2idx = {beh : idx for idx, beh in enumerate(behaviors)}

    # one_hot_vector = np.zeros(shape = (len(behaviors)))

    # one_hot_vector[behavior2idx[ground_truth]] = 1

    return behavior2idx[ground_truth]


def get_possible_behaviours(behavioural_map, current_node):
    #returns an one hot encoded vector denoting next possible actions
    '''
    behavioural_map - Adjacency List of the BM
    current_node - string
    Returns array of one hot encoded vectors of the possible behaviors
    '''
    behaviors = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl','end']
    behavior2idx = {beh : idx for idx, beh in enumerate(behaviors)}
    
    possible_behaviors = [edge[0] for edge in behavioural_map[current_node]]
    one_hot_vector = np.zeros(shape = (len(behaviors)))

    for behavior in possible_behaviors:
        one_hot_vector[behavior2idx[behavior]] = 1

    one_hot_vector[behavior2idx['end']] = 1
    return one_hot_vector # 1D numpy array of shape -> (total numbers of behaviors)

def predict_all_behs(Natural_lang_instr,start_node,beh_map, trained_pca, trained_model):
    out_beh = 'start'
    out_beh_arr = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl','end']
    beh2idx = {b:i for i,b in enumerate(out_beh_arr)}
    idx2beh = {i:b for i,b in enumerate(out_beh_arr)}
    #   print(idx2beh)
    conv_arr =[]
    conv_arr.append(start_node)

    print("converting into predictable format...")
    nl_seq = convert_nautural_sequence_into_embedding(emb_matrix,word2id,id2word,Natural_lang_instr)
    nl_seq = nl_seq.flatten()

    break_i = 0
    while out_beh!='end':
        # print(break_i)
        curr_node  = conv_arr[-1]
        encoded_seq = get_one_hot_encoded_sequence(token_to_index_array,conv_arr)
        encoded_seq = encoded_seq.flatten()

        try:
            poss_beh = get_possible_behaviours(beh_map,curr_node)
        except:
            break
        # poss_beh = poss_beh.flatten()

        data_row = np.concatenate([nl_seq, encoded_seq, poss_beh])

        data_row = data_row.reshape(1,-1)
        # print(data_row.shape)
        # print("transfrom...")
        reduced_vec = trained_pca.transform(data_row)
        reduced_vec = reduced_vec.reshape(-1,1)
        poss_beh = poss_beh.reshape(-1,1)
        model_input_vector = np.append(reduced_vec,poss_beh,0)
        model_input_vector = model_input_vector.reshape(1,-1)

        # print("predict...")
        out_beh_pred = trained_model.predict(model_input_vector)
        # print(out_beh_pred.shape)
        out_beh_pred = out_beh_pred.reshape(-1,1)

        # out_beh = np.argmax(out_beh_pred)
        # if (type(out_beh)!=int):
        #   usable_int = 0
        #   for i in out_beh:
        #     if(i==1):
        #       break
        #     else:
        #       usable_int+=1
        #   out_beh = usable_int

        # out_beh = out_beh_arr[out_beh]
        # print(out_beh)
        # max_vl = 0
        # for i range(poss_beh.shape[0]):
        #     if(poss_beh[i]==1):
        #         if(out_beh_pred[i]>out_beh_pred[max_vl]):
        #             max_vl = i


        ### Adding to the conv arr
        max_val = 0
        max_i = 0
        final_node = ''
        for edge in beh_map[curr_node]:
            if out_beh_pred[beh2idx[edge[0]]] > max_val:
                max_val = out_beh_pred[beh2idx[edge[0]]]
                max_i = beh2idx[edge[0]]
                final_node = edge[1]

        if max_val < out_beh_pred[12] and len(conv_arr) >= 3:
            print("End found!")
            out_beh = 'end'
        elif final_node in conv_arr:
            if conv_arr[-1] == padded_node:
                break
            else:
                print("Looping!")
                max_val = 0
                max_i = 0
                # final_node = ''
                for edge in beh_map[curr_node]:
                    if out_beh_pred[beh2idx[edge[0]]] > max_val and edge[1] != final_node:
                        max_val = out_beh_pred[beh2idx[edge[0]]]
                        max_i = beh2idx[edge[0]]
                        final_node = edge[1]

                out_beh = idx2beh[max_i]
                print(out_beh)
                conv_arr.append(out_beh)
                conv_arr.append(final_node)
        else:
            out_beh = idx2beh[max_i]
            print(out_beh)
            conv_arr.append(out_beh)
            conv_arr.append(final_node)

        # print(out_beh)
        
        if break_i == 20:
            print("uqwbqdjas;j")
            break

        break_i += 1

    return ' '.join(conv_arr)




# Natural_lang_instr = current_dataset[10][0]
# print(Natural_lang_instr)
# start_node = current_dataset[10][1][-1]
# print(current_dataset[5][1])
# beh_map = current_dataset[10][2]
# print(predict_all_behs(Natural_lang_instr, start_node, beh_map, emb_matrix, word2id, id2word, token_to_index_array, pcaModel, deepModel))