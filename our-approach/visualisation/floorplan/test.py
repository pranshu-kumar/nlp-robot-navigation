from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geopandas as gpd
import re
# floor_df = gpd.GeoDataFrame({
#     'room_name': ["Office-1"],
#     'geometry': [
#         Polygon([(1,1),(6,1),(6,6),(1,6)])
#         ],
# })

# graph_str = ""
behaviors = ['oor', 'ool', 'iol', 'ior', 'oio', 'cf', 'chs', 'lt', 'rt', 'sp', 'chr', 'chl']
_BEHAVIOR_RE = r"({})".format("|".join(behaviors))

with open("../../data/data.graph", 'r') as f:
    graph_str = f.readline()

graph_str = re.sub(r'\(.*?\)', '', graph_str)
rooms = set(re.findall(r'[O|R|K|B|H|L]-\d+', graph_str))
# print(graph_str)
triplet_encodings = graph_str.strip().split(';')
triplet_encodings = [t.lstrip().rstrip() for t in triplet_encodings[:-1]]

corridors = set()
oio_tups = [] ## Rooms in the tuples will be in front of each other 
for encoding in triplet_encodings:
    triplets = [r.strip() for r in re.split(_BEHAVIOR_RE, encoding)]
    b_attr = triplets[1]
    node_1 = triplets[0]
    node_2 = triplets[2]
    # print(node_1, node_2)
    # print(triplets)
    if b_attr == 'oio':
        oio_tups.append((node_1, node_2))

    if node_1.startswith('C'):
        corridors.add(node_1)

    if node_2.startswith('C'):
        corridors.add(node_2)

corridors = sorted(list(corridors)) 
'''
- Sorted corridors will give the ordering of the rooms on a side of the corridor
- using oio relations a corrdior could be constructed
'''
print(corridors)

    
# print(oio_tups)



# Annotating Room name on polygon object
def annotate_room_name(floor_df):
    floor_df['coords'] = floor_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    floor_df['coords'] = [coords[0] for coords in floor_df['coords']]
    return floor_df

# floor_df.boundary.plot()
# for idx, row in floor_df.iterrows():
#     plt.annotate(s=row['room_name'], xy=row['coords'],
#                 horizontalalignment='center')
# plt.show()

