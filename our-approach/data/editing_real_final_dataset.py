import pickle
f = open('final_dataset.pkl','rb')
dta = pickle.load(f)
# print(dta[0][0])
f1 = open('data.instruction','r')
lines_arr = f1.readlines()
new_data = []
for i,entry in enumerate(dta):
    entry[0] = lines_arr[i]
    new_data.append(entry)

new_dataset = []
for entry in new_data:
    for i,pathtilnow in enumerate(entry[1]):
        new_data_entry = []
        new_data_entry.append(entry[0])
        new_data_entry.append(entry[1][i])
        new_data_entry.append(entry[2])
        new_data_entry.append(entry[3][i])
        new_dataset.append(new_data_entry)

finalone = open('dataset_to_use.pkl','wb')
pickle.dump(new_dataset,finalone)
