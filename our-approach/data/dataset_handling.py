
# # import pickle
# # nfile = open('final_dataset.pkl','rb')
# # f1 = pickle.loads(nfile.read())
# # new_dataset = []
# # for data_entry in f1:
# #     for ind in range(len(data_entry[1])):
# #         new_data_entry = []
# #         new_data_entry.append(data_entry[0])
# #         new_data_entry.append(data_entry[1][ind])
# #         new_data_entry.append(data_entry[2])
# #         new_data_entry.append(data_entry[3][ind])
# #         new_dataset.append(new_data_entry)
# # real_final_dataset_file = open('real_final_dataset.pkl','wb')
# # pickle.dump(new_dataset,real_final_dataset_file)
# import pickle
# from nltk.stem import WordNetLemmatizer
# import re
# data_file = open('real_final_dataset.pkl','rb')
# dataset_object = pickle.load(data_file)
# data_batch = dataset_object[:10]

# print(data_batch)


# _WORD_SPLIT = re.compile(r"([.,!?\"':;()/-])")
# def preprocess_instruction(sentence):
#     # change "office-12" or "office12" to "office 12"
#     # change "12-office" or "12office" to "12 office"
#     _WORD_NO_SPACE_NUM_RE = r'([A-Za-z]+)\-?(\d+)'
#     _NUM_NO_SPACE_WORD_RE = r'(\d+)\-?([A-Za-z]+)'
#     new_str = re.sub(_WORD_NO_SPACE_NUM_RE, lambda m: m.group(1) + ' ' + m.group(2), sentence)
#     new_str = re.sub(_NUM_NO_SPACE_WORD_RE, lambda m: m.group(1) + ' ' + m.group(2), new_str)
#     lemma = WordNetLemmatizer()
#     # correct common typos.
#     correct_error_dic = {'rom': 'room', 'gout': 'go out', 'roo': 'room',
#                          'immeidately': 'immediately', 'halway': 'hallway',
#                          'office-o': 'office 0', 'hall-o': 'hall 0', 'pas': 'pass',
#                          'offic': 'office', 'leftt': 'left', 'iffice': 'office'}
#     for err_w in correct_error_dic:
#         find_w = ' ' + err_w + ' '
#         replace_w = ' ' + correct_error_dic[err_w] + ' '
#         new_str = new_str.replace(find_w, replace_w)
#     sen_list = []
#     # Lemmatize words
#     for word in new_str.split(' '):
#         try:
#             word = lemma.lemmatize(word)
#             if len(word) > 0 and word[-1] == '-':
#                 word = word[:-1]
#             if word:
#                 sen_list.append(word)
#         except UnicodeDecodeError:
#             continue
#             # print("unicode error ", word, new_str)
#     return sen_list


# def simple_tokenizer(sentence):
#     #removes characters like exclamation mark,comma,dot brackets etc and returns a tokenized array
#     words = []
#     prepocessed_sen_list = preprocess_instruction(sentence.strip())
#     for space_separated_fragment in prepocessed_sen_list:
#         words.extend(_WORD_SPLIT.split(space_separated_fragment))
#     return [w.lower() for w in words if w]

# import pickle 
# f = open('word2id.pkl','rb')
# a = pickle.load(f)

# print(a['dog'])