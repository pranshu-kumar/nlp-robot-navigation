import matplotlib.pyplot as plt
import networkx as nx
import re


behaviors = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl']
_BEHAVIOR_RE = r"({})".format("|".join(behaviors))

class Visualiser:
    def __init__(self, graph_str):
        self.graph_str = self.__preprocess_graph_str(graph_str)

    def plot_BM2graph(self):
        '''
        Function to plot the behavioral map into a graph
        NOTE: A corridor from either directions are treated as different nodes in the graph
        '''
        triplet_encodings = self.__get_triplet_encodings()
        
        G, all_nodes, all_edges = self.__get_graph_from_triplet_encodings(triplet_encodings)
        
        edge_labels_dict = self.__get_edge_labels_dict(all_nodes, all_edges)

        pos = nx.nx_agraph.graphviz_layout(G)
        plt.figure(figsize=(20,15))    
        nx.draw(G,pos,edge_color='black',width=1, linewidths=1, node_size=800, node_color='pink', connectionstyle='arc3, rad=0.0', alpha=0.9, labels={node:node for node in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_color='blue', font_size=12)
        plt.axis('off')
        plt.title("Behavioral Map Visualisation")
        # plt.savefig("bmap-1.png")
        plt.show()
    

    ################ PRIVATE FUNCTIONS ####################

    def __get_edge_labels_dict(self, all_nodes, all_edges):
        assert len(all_nodes) == len(all_edges)
        
        edge_labels_dict = {}
        for i in range(len(all_nodes)):
            tup = all_nodes[i]
            edge_labels_dict[(tup[0], tup[1])] = all_edges[i]

        return edge_labels_dict


    def __get_graph_from_triplet_encodings(triplet_encodings):
        G = nx.DiGraph()
        all_nodes = []
        all_edges = []

        for t in triplet_encodings:
            temp = t.split(" ")
            if len(temp) > 1:
                if len(temp) == 3:
                    G.add_node(temp[0])
                    G.add_node(temp[2])
                    G.add_edge(temp[0], temp[2])
                    all_edges.append(temp[1])
                    all_nodes.append((temp[0], temp[2]))
                else:
            #         print(t)
                    res = [r.strip() for r in re.split(_BEHAVIOR_RE, t)]
                    if res[0][0] == 'C' and res[2][0] == 'C':
                        rr1 = res[0].split(" ")
                        rr2 = res[2].split(" ")
                        G.add_node(rr1[0] + rr1[-1])
                        G.add_node(rr2[0] + rr2[-1])
                        G.add_edge(rr1[0] + rr1[-1], rr2[0] + rr2[-1])
                        all_nodes.append((rr1[0] + rr1[-1],rr2[0] + rr2[-1]))
                        all_edges.append(res[1])
                        
                    elif res[2][0] == 'C':
                        G.add_node(res[0])
                        rr = res[2].split(" ")
        #                 print(rr)
                        G.add_node(rr[0] + rr[-1])
                        G.add_edge(res[0], rr[0] + rr[-1])
                        all_nodes.append((res[0], rr[0] + rr[-1]))
                        all_edges.append(res[1])
                        
                    else:
                        G.add_node(res[2])
                        rr = res[0].split(" ")
        #                 print(rr)
                        G.add_node(rr[0] + rr[-1])
                        G.add_edge(res[2], rr[0] + rr[-1])
                        all_nodes.append((res[2], rr[0] + rr[-1]))
                        all_edges.append(res[1])
        
        return G, all_nodes, all_edges

    def __get_triplet_encodings(self):
        triplet_encodings = self.graph_str.strip().split(";")
        for i in range(len(triplet_encodings)):
            triplet_encodings[i] = triplet_encodings[i].rstrip().lstrip()

    def __preprocess_graph_str(self, graph_str):
        return re.sub(r'\(.*?\)', '', graph_str)

# tokens = graph_str.strip().split(";")
# for i in range(len(tokens)):
#     tokens[i] = tokens[i].rstrip().lstrip()s