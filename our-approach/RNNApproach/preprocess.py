from auxilary_functions import *
import pickle
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


_VOCAB_FILE_READ = open('../data/vocab.pkl', 'rb')
_DATASET_FILE_READ = open('../data/dataset_to_use.pkl', 'rb')
_EMBMAT_FILE_READ = open('../data/emb_matrix.pkl', 'rb')
_WORD2ID_FILE_READ = open('../data/word2id.pkl', 'rb')
_ID2WORD_FILE_READ = open('../data/id2word.pkl', 'rb')

data = pickle.load(_DATASET_FILE_READ)
token_to_index_array = pickle.load(_VOCAB_FILE_READ)
token_to_index_array[1]['end'] = 213
nl_emb_mat = pickle.load(_EMBMAT_FILE_READ)
word2id = pickle.load(_WORD2ID_FILE_READ)
id2word = pickle.load(_ID2WORD_FILE_READ)

up_emb_mat = get_up_emb_mat(token_to_index_array)
# print(up_emb_mat, up_emb_mat.shape)
# print(id2word[0])
def preprocess_data(data):

    if os.path.exists("RNN_data/nl_embeddings.pkl") and os.path.exists("RNN_data/until_path_vectors.pkl") and os.path.exists("RNN_data/ground_truth_vectors.pkl") and os.path.exists("RNN_data/possible_behaviors_embeddings.pkl"):
        nl_embeddings = pickle.load(open("RNN_data/nl_embeddings.pkl", 'rb'))
        until_path_vectors = pickle.load(open("RNN_data/until_path_vectors.pkl", 'rb'))
        ground_truth_vectors = pickle.load(open("RNN_data/ground_truth_vectors.pkl", 'rb'))
        possible_behaviors_embeddings = pickle.load(open("RNN_data/possible_behaviors_embeddings.pkl", 'rb'))
        
        return nl_embeddings, until_path_vectors, ground_truth_vectors, possible_behaviors_embeddings

    else:
        nl_indices = np.empty((82201, 265), dtype=np.float64)
        until_path_indices = np.empty((82201, 47), dtype=np.float64)
        ground_truth_vectors = []
        possible_behaviors_embeddings = []
        i = 0
        for d in tqdm(data):
            instruction = d[0]
            until_path = d[1]
            bm_adj_list = d[2]
            ground_truth = d[3]

            # print(until_path)

            # store NL sentence embedding
            tokens_indices = convert_nautural_sequence_into_indices(nl_emb_mat, word2id, id2word, instruction)
            nl_indices[i] = tokens_indices
            # get one hot vector of the until path seq
            until_path_indices_i = get_until_path_indices(token_to_index_array, until_path)
            # print(until_path_indices)
            until_path_indices[i] = until_path_indices_i

            # embed ground truth (one hot vector)
            ground_truth_vectors.append(embed_ground_truth(ground_truth))

            # get embedding of possible behaviors from BM adj list
            possible_behaviors_embeddings.append(get_possible_behaviours(bm_adj_list, until_path[-1]))
            
            i += 1
        # # Save data
        with open("nl_embeddings.pkl", 'wb') as f:
            pickle.dump(nl_embeddings, f)

        with open("until_path_vectors.pkl", 'wb') as f:
            pickle.dump(until_path_vectors, f)

        with open("ground_truth_vectors.pkl", 'wb') as f:
            pickle.dump(ground_truth_vectors, f)

        with open("possible_behaviors_embeddings.pkl", 'wb') as f:
            pickle.dump(possible_behaviors_embeddings, f)

        ground_truth_vectors = np.stack(ground_truth_vectors, axis=0)

        return nl_indices, until_path_indices, ground_truth_vectors, possible_behaviors_embeddings

# preprocess_data(data)
