from django.shortcuts import render
from .forms import TranslationForm
from .predict import predict_all_behs
from keras.models import load_model
import joblib
import os
import networkx as nx
import matplotlib.pyplot as plt
import plotly
from io import StringIO

# Create your views here.
def translate_instruction(request):
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            return HTTpResponseRedirect('/robottranslation/')
    else:
        form = TranslationForm()

    return render(request, 'translator.html', {'form':form})

def show_translation(request):
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            # Fetch form data
            nl_instruction = form['nl_instruction'].value()
            modified_map = form['b_map'].value()

            b_map_adj_list = get_adj_list_of_bm(modified_map)
            start_node = form['start_node'].value()
            
            # Load models
            print("Loading models...")
            model_path = 'Django usable model'
            deepModel = load_model(model_path)
            pcaModel = joblib.load(open(os.path.dirname(os.path.realpath(__file__)) + '/data/reduction_model.sav', "rb"))
            print("Models loaded!")

            # Predict
            robot_lang = predict_all_behs(nl_instruction, start_node, b_map_adj_list, pcaModel, deepModel)

            G, all_nodes, all_edges = get_networkx_graph(b_map_adj_list)
            print(len(all_nodes), len(all_edges))
            edge_labels_dict = get_edge_labels_dict(all_nodes, all_edges)

            b_map_graph_div = get_b_map_div(G, edge_labels_dict)

    else:
        form = TranslationForm()
    return render(request, 'robottranslation.html', {
        'nl_instruction':nl_instruction,
        'start_node':start_node,
        'robot_lang':robot_lang,
        'b_map_graph_div':b_map_graph_div
        })

def get_adj_list_of_bm(modified_map):
    # for modified_map in modified_beh_maps:
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
    return graph_behmap

def get_networkx_graph(b_map):
    # print(b_map)
    G = nx.DiGraph()
    all_nodes = []
    all_edges = []
    for node in b_map:
        for edge in b_map[node]:
            all_edges.append(edge[0])
            all_nodes.append((node, edge[1]))

            G.add_node(node)
            G.add_node(edge[1])
            G.add_edge(node, edge[1])

    # print(all_edges)
    return G, all_nodes, all_edges


def get_edge_labels_dict(all_nodes, all_edges):
        assert len(all_nodes) == len(all_edges)
        
        edge_labels_dict = {}
        for i in range(len(all_nodes)):
            tup = all_nodes[i]
            edge_labels_dict[(tup[0], tup[1])] = all_edges[i]

        return edge_labels_dict

def get_b_map_div(G, edge_labels_dict):
    pos = nx.nx_agraph.graphviz_layout(G)
    fig = plt.figure(figsize=(20,15))    
    nx.draw(G,pos,edge_color='black',width=1, linewidths=1, node_size=800, node_color='pink', connectionstyle='arc3, rad=0.0', alpha=0.9, labels={node:node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_color='blue', font_size=12)
    plt.axis('off')
    plt.title("Behavioral Map Visualisation")

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data
    