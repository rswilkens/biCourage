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


def load_graphs(dataset_path,encoding="onehot", edge_type="ngram3", lang = "en", min_sent_len = 200, nun_classes=2):
  # if ignore_graph:
  #   return __load_X(dataset_path=dataset_path,encoding=encoding, edge_type=edge_type, lang = "en", minsetnlen = min_sent_len)
  # else:
  if type(edge_type) == str:
    return __load_graphs(dataset_path=dataset_path, encoding=encoding, edge_type=edge_type[0], lang=lang, minsetnlen = min_sent_len,nun_classes=nun_classes)
  else:
    assert len(edge_type) == len(set(edge_type)) # no duplicated values
    maxlen = -1
    vals = []
    for edge_tp in edge_type:
      lst1, maxlen1, ids1 = __load_graphs(dataset_path=dataset_path, encoding=encoding, edge_type=edge_tp, lang=lang, minsetnlen = min_sent_len, return_ids = True, nun_classes=nun_classes)
      if 'syntactic' in edge_tp:
          # print('--------------------->syntactic',maxlen1,"->",maxlen1 - 1)
          maxlen1 -= 1 # syn == ngram+1
      if maxlen < 0:
        maxlen = maxlen1
      else:        
        # print("maxlen1,maxlen:", maxlen1,maxlen)
        assert max(maxlen1, maxlen) == min(maxlen1, maxlen) 
      vals.append([lst1, ids1])
    return __join_gragph_list(vals), maxlen

  # if "_" in edge_type:
  #   edge_type1, edge_type2 = edge_type.split("_")
  #   print("edge_type.split",  edge_type.split("_") )
  #   lst1, maxlen1, ids1 = __load_graphs(dataset_path=dataset_path, encoding=encoding, edge_type=edge_type1, lang=lang, minsetnlen = min_sent_len, return_ids = True, nun_classes=nun_classes)
  #   lst2, maxlen2, ids2 = __load_graphs(dataset_path=dataset_path, encoding=encoding, edge_type=edge_type2, lang=lang, minsetnlen = min_sent_len, return_ids = True, nun_classes=nun_classes)
  #   # print(len(lst1), maxlen1, len(ids1))
  #   # print(len(lst2), maxlen2, len(ids2))
  #   assert max(maxlen1, maxlen2)-1 == min(maxlen1, maxlen2) # syn == ngram+1 
  #   return __join_gragph_list(lst1,lst2, ids1, ids2), maxlen1
  # else:
  #   return __load_graphs(dataset_path=dataset_path, encoding=encoding, edge_type=edge_type, lang=lang, minsetnlen = min_sent_len,nun_classes=nun_classes)

# def __join_gragph_list(lst1,lst2, ids1, ids2):
#   if len(ids1) == len(ids2):
#     ret = []
#     for g1, g2, i1, i2 in zip(lst1,lst2, ids1, ids2):
#       assert i1 == i2
#       ret.append([g1,g2])
#     return ret
#   else:
#     ret = []
#     ids = set(ids1) | set(ids2)
#     for id in ids:
#       if id in ids1 and id in ids2:
#         index1 = ids1.index(id)
#         index2 = ids2.index(id)
#         g1 = lst1[index1]
#         g2 = lst2[index2]
#         ret.append([g1,g2])
#     return ret
def __join_gragph_list(vals):
  all_ids = set()
  iters_lst = []
  iters_ids = []
  for l, ids in vals:
    all_ids = all_ids | set(ids)
    iters_lst.append(iter(l))
    iters_ids.append(iter(ids))
  for _, ids in vals:
    if len(ids) != all_ids:
      print('!!!!!\tmissing values', all_ids - set(ids))
  all_ids = vals[0][1] # use the IDs from first graph as reference index
  ret = []
  for ids in all_ids:
    val = []
    for l, i in zip(iters_lst, iters_ids):
      assert next(i) == ids
      val.append(next(l))
    # print("DEBUG:", ids, len(val), len(iters_lst), len(iters_ids))
    ret.append(val)
  return ret

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
  ##############################################
  if len(edges) < len(ids):
    print("DEBUG START: id removed:", len(cls), len(ids), len(edges))
    for id in ids:
      if id not in edges: # if id not in ids:
        ids.remove(id)
        print("id removed:", id, len(cls), len(ids), len(edges))
  ##############################################
  assert len(edges) >= len(ids)
  print("...embeddings",dataset_path + "_"+ encoding +"_"+lang+".pkl")
  embeddings = pickle.load(open(dataset_path + "_"+ encoding +"_"+lang+".pkl", 'rb'))
  assert len(embeddings) >= len(ids)
  graphs = []
  maxlen = 0
  final_ids = []
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
    # print(cls[id], y, positive)
    # input("----")
    assert y != None
    cls.pop(id)
    embeddings.pop(id)
    maxlen = max(maxlen,len(x))
    assert len(x) > 0
    # print("edge_type",edge_type, len(x), len(set(edge_index[0]) | set(edge_index[1])), len(ids))
    #assert len(x) == len(set(edge_index[0]) | set(edge_index[1]))
    assert len(edge_index[0]) == len(edge_weight)
    if len(set(edge_index[0]) | set(edge_index[1])) != len(x):
         print(len(set(edge_index[0])), len(set(edge_index[1])))
         print(len(set(edge_index[0]) | set(edge_index[1])), len(x) )
    # assert len(set(edge_index[0]) | set(edge_index[1])) == len(x)
    if len(set(edge_index[0]) | set(edge_index[1])) == len(x):
      g = IdentifiedGraph(
                id=id,
                x=np.asarray(x),
                edge_index=np.asarray(edge_index),
                edge_weight=np.asarray(edge_weight),
                y=[y]
            )
      final_ids.append(id)
      graphs.append(g)
#     assert len(set(edge_index[0]) | set(edge_index[1])) == len(x)
#     g = IdentifiedGraph(
#               id=id,
#               x=np.asarray(x),
#               edge_index=np.asarray(edge_index),
#               edge_weight=np.asarray(edge_weight),
#               y=[y]
#           )
#     graphs.append(g)
#     final_ids.append(id)
  if return_ids:
    return graphs, maxlen, final_ids
  return graphs, maxlen



