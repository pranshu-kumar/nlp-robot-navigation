import torch
import numpy as np

def get_one_hot_encoded_sequence(token_to_index_array, array_for_conversion):
    #converts a robot language sequence into an onehot encoded sequence
    '''
    token_to_index_array - RL token2id
    array_for_conversion - subset path
    Return array of one hot encoded vectors of the subset paths
    '''
    pass

def convert_nautural_sequence_into_embedding(emb_matrix, word2id, id2word, natural_language_sequence):
    #returns an embedding vector corresponding to a natural language sequence
    '''
    First tokenise using simple_tokenizer  
    returns embedding matrix of the NL instruction
    '''
    pass

def simple_tokenizer(natural_langauge_sequence):
    #removes characters like exclamation mark,comma,dot brackets etc and returns a tokenized array
    pass

def get_possible_behaviours(behavioural_map, current_node):
    #returns an one hot encoded vector denoting next possible actions
    '''
    behavioural_map - Adjacency List of the BM
    current_node - string
    Returns array of one hot encoded vectors of the possible behaviors
    '''
    pass

def convert_entry_into_trainable_format(dataset_entry, emb_matrix, word2id, id2word, token_to_index, token_array_to_convert, behaviour_array):
    '''
    Takes in dataset entry : NL instr,robot path,possibilities,behav_map_in_graph,output
    and returns it into an encoded format or the trainable format
    '''
    pass
