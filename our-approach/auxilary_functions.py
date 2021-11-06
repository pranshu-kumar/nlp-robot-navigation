import torch
import numpy as np
import pandas as pd

def get_one_hot_encoded_sequence(token_to_index_array,array_for_conversion):
    #converts a robot language sequence into an onehot encoded sequence
    pass

def convert_nautural_sequence_into_embedding(embmatrix,vec2id,id2vec,natural_language_sequence):
    #returns an embedding vector corresponding to a natural language sequence
    pass

def tokenizer_and_special_char_remover(natural_langauge_sequence):
    #removes characters like exclamation mark,comma,dot brackets etc and returns a tokenized array
    pass

def get_possible_behaviours(behavioural_map,current_node):
    #returns an one hot encoded vector denoting next possible actions
    pass

def convert_entry_into_trainable_format(dataset_entry,embmatrix,vec2id,id2vec,token_to_index,token_array_to_convert,behaviour_array):
    '''
    Takes in dataset entry : NL instr,robot path,possibilities,behav_map_in_graph,output
    and returns it into an encoded format or the trainable format
    '''
    pass
