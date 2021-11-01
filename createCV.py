from sklearn.model_selection import StratifiedKFold
import pickle
import numpy as np
import tensorflow as tf
import tf_geometric as tfg
from tf_geometric.utils.graph_utils import convert_edge_to_directed, compute_edge_mask_by_node_index
from tf_geometric.utils.tf_sparse_utils import sparse_tensor_gather_sub
from tf_geometric.utils.union_utils import union_len, convert_union_to_numpy



class IdentifiedGraph(tfg.Graph):
  def __init__(self, id, x, edge_index, y, edge_weight=None):
    self.id = id
    self.x = tfg.Graph.cast_x(x)
    self.edge_index = tfg.Graph.cast_edge_index(edge_index)
    self.y = tfg.Graph.cast_y(y)
    self.cache = {}
    if edge_weight is not None:
        self.edge_weight = self.cast_edge_weight(edge_weight)
    else:
        self.edge_weight = np.full([self.num_edges], 1.0, dtype=np.float32)
        if tf.is_tensor(self.x):
            self.edge_weight = tf.convert_to_tensor(self.edge_weight)
  @property
  def get_id(self):
    return self.id


def __load_graphs(dataset_path,encoding="onehot", edge_type="ngram3", lang = "en", minsetnlen = 200, return_ids = False, nun_classes=2, positive="hate"):
  print("...open",dataset_path + ".pkl")
  full_ds, _ = pickle.load(open(dataset_path + ".pkl", 'rb'))
  ids = []  
  for item in full_ds:
    if lang in item["lang"]: 
      # if item["id"] in ids: # # repetead ID
      #   continue
      ids.append(item["id"])
  print(dataset_path, "dataset len", len(ids), "from", len(full_ds), ids[:10])
  #input()
  assert len(ids) > 0
  print("...edges",dataset_path + "_"+ edge_type +".pkl")
  full_edges = pickle.load(open(dataset_path + "_"+ edge_type +".pkl", 'rb'))
  edges = {}
  cls = {}
  for edge in full_edges:
    if edge["id"] in ids:
      edges[edge["id"]] = {"edge_index":edge["edge_index"], "edge_weight":edge["edge_weight"]}
      cls[edge["id"]] = edge["cls"]
  assert len(edges) >= len(ids)
  print("...embeddings",dataset_path + "_"+ encoding +"_"+lang+".pkl")
  embeddings = pickle.load(open(dataset_path + "_"+ encoding +"_"+lang+".pkl", 'rb'))
  assert len(embeddings) >= len(ids)
  graphs = []
  maxlen = 0
  for id in ids:
    x = embeddings[id]
    if "syntactic" in edge_type:
      zero = np.mean(x, axis=0)
      zero = zero.reshape((1, zero.shape[0]))
      x = np.concatenate((zero, x), axis=0) 
    edge_index = edges[id]["edge_index"]
    edge_weight = (np.absolute(edges[id]["edge_weight"])+0.0000000001) / len(x)
    y = None
    if nun_classes==2:
      if cls[id].lower() == positive.lower():
        y = 1
      else:
        y = 0
    assert y != None
    cls.pop(id)
    embeddings.pop(id)
    maxlen = max(maxlen,len(x))
    assert len(x) > 0
    assert len(edge_index[0]) == len(edge_weight)
    if len(set(edge_index[0]) | set(edge_index[1])) != len(x):
         print(len(set(edge_index[0])), len(set(edge_index[1])))
         print(len(set(edge_index[0]) | set(edge_index[1])), len(x) )
    assert len(set(edge_index[0]) | set(edge_index[1])) == len(x)
    g = IdentifiedGraph(
              id=id,
              x=np.asarray(x),
              edge_index=np.asarray(edge_index),
              edge_weight=np.asarray(edge_weight),
              y=[y]
          )
    graphs.append(g)
  if return_ids:
    return graphs, maxlen, ids
  return graphs, maxlen



def createCV(dataset_path, lang):
	graphs, maxlen = __load_graphs(dataset_path=dataset_path, encoding="wordembedding", edge_type="ngram3", lang=lang, nun_classes=2) 
	y = [val.y[0] for val in graphs]
	skf = StratifiedKFold(n_splits=10, shuffle=True)
	with open(dataset_path + "_cv_"+ lang  +".txt", "w") as output_file:
		for fold_index, (train_index, test_index) in enumerate(skf.split(graphs, y)):
			for index in test_index:
				# print(str(fold_index) + "\t" + str(graphs[index].get_id) + "\t" + str(graphs[index].y[0]))
				output_file.write(str(fold_index) + "\t" + str(graphs[index].get_id) + "\t" + str(graphs[index].y[0]) + "\n")


createCV(dataset_path = "../pkl/only_exist.csv", lang = "en")
createCV(dataset_path = "../pkl/only_exist.csv", lang = "es")

createCV(dataset_path = "../pkl/only_hasoc2019.csv", lang = "en")
createCV(dataset_path = "../pkl/only_hasoc2019.csv", lang = "de")

createCV(dataset_path = "../pkl/only_ami2018elg.csv", lang = "en")
createCV(dataset_path = "../pkl/only_ami2018elg.csv", lang = "it")

createCV(dataset_path = "../pkl/only_hateval.csv", lang = "en")
createCV(dataset_path = "../pkl/only_hateval.csv", lang = "es")

