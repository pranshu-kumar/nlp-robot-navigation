import re
import pickle
# import nltk 
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import numpy as np
from numpy.core.fromnumeric import shape
dataset_file = open('dataset_to_use.pkl','rb')
_EMBMAT_FILE_READ = open('emb_matrix.pkl', 'rb')
_WORD2ID_FILE_READ = open('word2id.pkl', 'rb')
_ID2WORD_FILE_READ = open('id2word.pkl', 'rb')
_VOCAB_FILE_READ = open('vocab.pkl', 'rb')
emb_matrix = pickle.load(_EMBMAT_FILE_READ)
word2id = pickle.load(_WORD2ID_FILE_READ)
id2word = pickle.load(_ID2WORD_FILE_READ)
token_to_index_array = pickle.load(_VOCAB_FILE_READ)

current_dataset = pickle.load(dataset_file)
# emb_matrix,word2id,id2word,vocab_pkl,

_WORD_SPLIT = re.compile(r"([.,!?\"':;()/-])")
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
    max_length = 100


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




def further_modify_dataset(current_dataset):
    new_dataset = []
    for entry in current_dataset:
        # print(entry[0])
        natural_lang_embedded = convert_nautural_sequence_into_embedding(emb_matrix,word2id,id2word,entry[0]) # (265, 100)
        # print(natural_lang_embedded.sum())
        natural_lang_embedded = natural_lang_embedded.flatten()
        encoded_till_now_path = get_one_hot_encoded_sequence(token_to_index_array,entry[1]) # (25, 213)
        encoded_till_now_path = encoded_till_now_path.flatten()
        # print(encoded_till_now_path.shape)
        subset_path = entry[1]
        curr_node = subset_path[-1]
        b_map = entry[2]

        possible_behaviors = get_possible_behaviours(b_map, curr_node)
        
        data_row = np.concatenate([natural_lang_embedded, encoded_till_now_path, possible_behaviors])
        
        new_dataset.append(data_row)
        outtoken = entry[3]
    
    print(len(new_dataset))
    print(new_dataset[0].shape)

further_modify_dataset(current_dataset)
        



def get_padded_and_encoded_batch_entries(batch_size,dataset):
    #embedd the entire dataset
    #padd the entire dataset
    #batch the entire dataset
    pass


