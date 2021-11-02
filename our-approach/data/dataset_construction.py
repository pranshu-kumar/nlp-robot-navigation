
import re
import pickle
beh_maps_pointer = open('data.graph','r')
beh_maps_repeatition = list(beh_maps_pointer.readlines())
beh_maps_list = list(set(beh_maps_repeatition))
beh_maps_list.sort()
#imp_regex = \(.*?\)

behmap_to_index = {}
for i in range(len(beh_maps_list)):
    behmap_to_index[beh_maps_list[i]] = i

modified_beh_maps = []
correlation_maps_list = []
for map_index in range(len(beh_maps_list)):
    map = beh_maps_list[map_index]
    corr_list = set(re.findall('C-\d.*?[NSWE]',map))
    corr_list = list(corr_list)
    corr_list.sort()
    corr_map = {}
    for c in corr_list:
        lst = c.split()
        if(lst[0].strip() not in list(corr_map.keys())):
            corr_map[lst[0].strip()] = []
        corr_map[lst[0].strip()].append(c)

    for item in list(corr_map.items()):
        for num in range(len(item[1])):
            map = map.replace(item[1][num],item[0]+str(num))
    
    correlation_maps_list.append(corr_map)
    modified_beh_maps.append(map)



graph_encoded_behmaps = []

for modified_map in modified_beh_maps:
    mmap = modified_map.split(';')
    graph_behmap = {}
    for elem in mmap:
        elem = elem.strip()
        if(len(elem.split())==3):
            elem_arr = elem.split()
            el0 = elem_arr[0].strip()
            el1 = elem_arr[1].strip()
            el2 = elem_arr[2].strip()
            if(el0 not in list(graph_behmap.keys())):
                graph_behmap[el0] = []
            graph_behmap[el0].append([el1,el2])
    graph_encoded_behmaps.append(graph_behmap)

# print(graph_encoded_behmaps[behmap_to_index[beh_maps_repeatition[0]]]['K-0'])
# print(correlation_maps_list[behmap_to_index[beh_maps_repeatition[0]]])
## generating a file
new_answer_file = open('new_data.answer','w+')

orig_answer_file = open('data.answer','r')
orig_answer_instructions = list(orig_answer_file.readlines())

orig_answer_instructions = [i.strip() for i in orig_answer_instructions]
modified_instructions = []
for orig_instruction_index in range(len(orig_answer_instructions)):
    orig_instruction = orig_answer_instructions[orig_instruction_index]
    orig_instruction_splitted = orig_instruction.split()
    behmap_for_orig = graph_encoded_behmaps[behmap_to_index[beh_maps_repeatition[orig_instruction_index]]]
    new_instr = []
    new_instr.append(orig_instruction_splitted[0])
    for beh_ind in range(1,len(orig_instruction_splitted)):
        # print("inside loop",beh_ind)
        # print(new_instr)
        if('C-' not in orig_instruction_splitted[beh_ind]):
            # print('straight in',orig_instruction_splitted[beh_ind])
            new_instr.append(orig_instruction_splitted[beh_ind])
        else:
            init_nd = new_instr[-2]
            init_ins = new_instr[-1]
            # print(init_nd)
            # print(init_ins)
            
           
            tups = behmap_for_orig[init_nd]
            # print(tups)
            
            for tup in tups:
                if(init_ins==list(tup)[0] and 'C-' in list(tup)[1]):
                    new_instr.append(list(tup)[1])
                    # print(list(tup)[1])
                    break
        
    string_new_str = " ".join(new_instr)
    # print(string_new_str)
    modified_instructions.append(string_new_str)
# new_answer_file.writelines(modified_instructions)
for line in modified_instructions:
    new_answer_file.write(line)
    new_answer_file.write('\n')

def return_until_path_array(ans_string):
    ans_string = ans_string.strip()
    ans_string = ans_string.split()
    ans_string_arr = [i.strip()for i in ans_string]
    answer_arr = []
    for nn in range(len(ans_string_arr)):
        if(nn!=0 and nn%2==1):
            answer_arr.append(ans_string_arr[:nn])
    answer_arr.append(ans_string_arr)
    return answer_arr

def return_outtokens(ans_string):
    ans_string = ans_string.strip()
    ans_string = ans_string.split()
    ans_string_arr = [i.strip()for i in ans_string]
    answer_arr = []
    for nn in range(len(ans_string_arr)):
        if(nn!=0 and nn%2==1):
            answer_arr.append(ans_string_arr[nn])
    answer_arr.append('end')
    return answer_arr

dataset = []
# beh_maps_repeatition = list(beh_maps_repeatition)

natural_lang_instr_file = open('data.instruction','r')

natural_instructions = list(natural_lang_instr_file.readlines())
for final_index in range(len(natural_instructions)):
    data_entry = []
    data_entry.append(natural_instructions[final_index])
    data_entry.append(return_until_path_array(modified_instructions[final_index]))
    data_entry.append(graph_encoded_behmaps[behmap_to_index[beh_maps_repeatition[final_index]]])
    data_entry.append(return_outtokens(modified_instructions[final_index]))
    dataset.append(data_entry)


pkl_file = open('final_dataset.pkl','wb')
pickle.dump(dataset,pkl_file)




