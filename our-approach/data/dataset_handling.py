
import pickle
nfile = open('final_dataset.pkl','rb')
f1 = pickle.loads(nfile.read())
new_dataset = []
for data_entry in f1:
    for ind in range(len(data_entry[1])):
        new_data_entry = []
        new_data_entry.append(data_entry[0])
        new_data_entry.append(data_entry[1][ind])
        new_data_entry.append(data_entry[2])
        new_data_entry.append(data_entry[3][ind])
        new_dataset.append(new_data_entry)
real_final_dataset_file = open('real_final_dataset.pkl','wb')
pickle.dump(new_dataset,real_final_dataset_file)
