import torch
import numpy as np
# from sklearn.preprocessing import OneHotEncoder
import pickle
# from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


# ######################## For testing ################################
# _VOCAB_FILE_READ = open('../data/vocab.pkl', 'rb')
# _FINAL_DATASET_FILE_READ = open('../data/final_dataset.pkl', 'rb')
# _EMBMAT_FILE_READ = open('../data/emb_matrix.pkl', 'rb')
# _WORD2ID_FILE_READ = open('../data/word2id.pkl', 'rb')
# _ID2WORD_FILE_READ = open('../data/id2word.pkl', 'rb')
# ####################################################################


_WORD_SPLIT = re.compile(r"([.,!?\"':;()/-])")


def get_up_emb_mat(token_to_index_array):
    
    token2id = token_to_index_array[1]
    all_tokens = token_to_index_array[0]

    max_length = 47
    up_emb_mat = np.eye(N=len(all_tokens), M=max_length)

    return up_emb_mat




def get_until_path_indices(token_to_index_array, until_path):
    
    all_tokens = token_to_index_array[0]
    token2id = token_to_index_array[1]
    id2token = {token2id[token]:token for token in all_tokens}
    
    until_path_indices = []
    for token in until_path:
        until_path_indices.append(token2id[token])

    max_length = 47
    padding = [token2id['end']]*(max_length-len(until_path_indices))

    until_path_indices.extend(padding)

    return np.array(until_path_indices)



def get_one_hot_encoded_sequence(token_to_index_array, array_for_conversion):
    #converts a robot language sequence into an onehot encoded sequence
    '''
    token_to_index_array - RL token2id
    array_for_conversion - subset path
    Return array of one hot encoded vectors of the subset paths
    '''
    all_tokens = token_to_index_array[0]
    tokens2id = token_to_index_array[1]

    
    # Define a max length -> padding size
    max_length = 20


    one_hot_vector = np.zeros(shape = (len(array_for_conversion), max_length, len(all_tokens)))
    
    # One hot encoding main 
    for i,  subset in enumerate(array_for_conversion):
        for j, token in enumerate(subset):
            
            idx = tokens2id[token]
            one_hot_vector[i, j, idx] = 1

    return one_hot_vector # numpy array of shape (number of subsets in array_for_conversion, max_length, length of all tokens in BM)
        
def convert_nautural_sequence_into_embedding(emb_matrix, word2id, id2word, natural_language_sequence):
    #returns an embedding vector corresponding to a natural language sequence
    '''
    First tokenise using simple_tokenizer  
    returns embedding matrix of the NL instruction
    '''
    NL_tokenized = simple_tokenizer(natural_language_sequence)
    
    token_ids = [word2id[token] for token in NL_tokenized]
    # print(emb_matrix.shape)
    nl_emb_matrix = [emb_matrix[idx] for idx in token_ids]

    nl_emb_matrix = np.stack(nl_emb_matrix, axis=0) 

    ## @Prasad this embedding matrix would need padding as well, I think - Pranshu
    
    return nl_emb_matrix # numpy 2D array with shape -> (number of tokens in NL, GloVE dimensions)



def convert_nautural_sequence_into_indices(emb_matrix, word2id, id2word, natural_language_sequence):
    #returns an embedding vector corresponding to a natural language sequence
    '''
    First tokenise using simple_tokenizer  
    returns embedding matrix of the NL instruction
    '''
    NL_tokenized = simple_tokenizer(natural_language_sequence)
    
    token_ids = [word2id[token] for token in NL_tokenized]
    # print(emb_matrix.shape)
    max_len = 265

    padding = [0]*(max_len-len(token_ids))
    token_ids.extend(padding)

    return token_ids


def simple_tokenizer(sentence):
    #removes characters like exclamation mark,comma,dot brackets etc and returns a tokenized array
    words = []
    prepocessed_sen_list = preprocess_instruction(sentence.strip())
    for space_separated_fragment in prepocessed_sen_list:
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]

## Added the same preprocess algo used in the baseline to save time. Works fine
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
                         'offic': 'office', 'leftt': 'left', 'iffice': 'office'}
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
    one_hot_vector = np.zeros(shape = (len(possible_behaviors), len(behaviors)))

    for i, behavior in enumerate(possible_behaviors):
        one_hot_vector[i,behavior2idx[behavior]] = 1

    return one_hot_vector # 2D numpy array of shape -> (number of possible behaviors, total numbers of behaviors)
    

def embed_ground_truth(ground_truth):
    behaviors = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl','end']
    behavior2idx = {beh : idx for idx, beh in enumerate(behaviors)}

    one_hot_vector = np.zeros(shape = (len(behaviors)))

    one_hot_vector[behavior2idx[ground_truth]] = 1

    return one_hot_vector

def convert_entry_into_trainable_format(dataset_entry, emb_matrix, word2id, id2word, token_to_index, token_array_to_convert, behaviour_array):
    '''
    Takes in dataset entry : NL instr,robot path,possibilities,behav_map_in_graph,output
    and returns it into an encoded format or the trainable format
    '''
    pass




## For testing 
# token_to_index_array = pickle.load(_VOCAB_FILE_READ)
# data = pickle.load(_FINAL_DATASET_FILE_READ)
# natural_language_sequence = data[0][0]
# BM_graph = data[0][2]
# emb_matrix = pickle.load(_EMBMAT_FILE_READ)
# word2id = pickle.load(_WORD2ID_FILE_READ)
# id2word = pickle.load(_ID2WORD_FILE_READ)
