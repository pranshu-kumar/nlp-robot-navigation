import networkx as nx
import matplotlib.pyplot as plt



''' *** Behaviors ***
oo<d> Go out of the current place and turn <d>
io<d> Turn <d> and enter the place straight ahead
oio Exit current place and enter straight ahead
<d>t Turn <d> at the intersection
cf Follow (or go straight down) the corridor
sp go straight at a t intersection
ch<d> Cross the hall and turn <d>
'''


class Visualizer:
    def __init__(self, filename):
        # filename -> name of the file where the input graph is
        # GraphString -> Input string of the graph
        self.graphString = self.__get_graph_string(filename)
        
        # define behavior dictionary
        self.behavior_dict = {
            'oor' : "Go out of the current place and turn right",
            'ool' : "Go out of the current place and turn left",
            'iol' : "Turn left and enter the place straight ahead",
            'ior' : "Turn right and enter the place straight ahead",
            'oio' : "Exit current place and enter straight ahead",
            'lt' : "Turn left at the intersection",
            'rt' : "Turn right at the intersection",
            'cf' : "Follow (or go straight down) the corridor",
            'sp' : "go straight at a T intersection",
            'chr': "Cross the hall and turn right",
            'chl': "Cross the hall and turn left"
            }

        # define landmarks
        self.landmarks = {
            "o" : "Office",
            "c" : "Corridor",
            "r" : "Room",
            'b' : "Bathroom",
            'l' : "Lab"
        }

        # Define directed graph
        self.G = nx.DiGraph()


    def plot_behavioral_graph(self):
        # Function to plot the BG
        # Using the library Networkx
        behaviors_array = self.__parse_graph()[:-1]
        
        # print(behaviors_array) 
        all_nodes = []
        all_edges = []
        for behavior in behaviors_array :
            elements = behavior.split(' ')
            nodes = []
            edge_label = ''
            for element in elements :
                if element not in self.behavior_dict :
                    if '-' in element : 
                        temp = element.split('-')
                        landmark = temp[0]
                        landmark_number = temp[-1]
                        
                        landmark = self.landmarks[landmark]
                        landmark = landmark + '-' + landmark_number
                        
                        # add node in the graph
                        self.G.add_node(landmark)

                        # add node in nodes
                        nodes.append(landmark)
                else : 
                    edge_label = element
                    # print(edge_label)
            self.G.add_edge(nodes[0], nodes[1])
            all_nodes.append((nodes[0], nodes[1]))
            all_edges.append(edge_label)
            
        edge_labels_dict = self.__get_edge_labels(all_nodes, all_edges)
        
        pos = nx.nx_agraph.graphviz_layout(self.G)
        plt.figure()    
        nx.draw(self.G,pos,edge_color='black',width=1, linewidths=1, node_size=5000, node_color='pink', connectionstyle='arc3, rad=0.2', alpha=0.9, labels={node:node for node in self.G.nodes()})
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels_dict, font_color='blue', font_size=12)
        plt.axis('off')
        plt.show()

    
    #~~~~~~~~~~ private functions ~~~~~~~~~~~~~#
    def __get_graph_string(self, filename):
        with open(filename, 'r') as f:
            graphString = f.read()

        return graphString

    def __parse_graph(self):
        # Function to parse the graphString input
        # Returns -> 
        behaviors_array = self.graphString.split(';')
        behaviors_array = self.__clean_behaviors_array(behaviors_array)
        return behaviors_array
        
    def __clean_behaviors_array(self, behaviors_array):
        return [b.strip() for b in behaviors_array if len(b) != 0]
    
    def __get_edge_labels(self, all_nodes, all_edges):
        
        assert len(all_nodes) == len(all_edges)
        
        edge_labels_dict = {}
        for i in range(len(all_nodes)):
            tup = all_nodes[i]
            edge_labels_dict[(tup[0], tup[1])] = all_edges[i]
        
        # print(edge_labels_dict)
        return edge_labels_dict



v = Visualizer("g1.graph")
v.plot_behavioral_graph()
