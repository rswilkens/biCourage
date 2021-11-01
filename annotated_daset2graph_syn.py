# -*- coding: utf-8 -*-
"""step3a_daset2graph-bertSyn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zRsZ9awIsINC4UNmtci14qbcsjU13bF0
"""

# !pip install transformers

# !pip install networkx

from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import numpy as np
import random
from nltk.util import ngrams
# import tf_geometric as tfg
import networkx as nx
from transformers import BertTokenizer
import os.path

base_dir = "/content/drive/MyDrive/Courage_GCN_HS/dataset/"

tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")

"""#text2graph

"""

def __follow(node, doc_edge_index, path = None):
  starts = [index for index, n in enumerate(doc_edge_index[0]) if n == node]
  if path == None:
    path = set()
  path.add(node)
  for start in starts:
    target = doc_edge_index[1][start]
    if target in path:
      continue
    path = path | __follow(target,doc_edge_index, path)
  # print("following", node, starts, path)
  # input()
  return path

def __add_discoannected_to_the_root(doc_edge_index, doc_edge_weight):
  ok = False
  while not ok:
    ok = True
    nodes = set(doc_edge_index[0])
    nodes.remove(0)
    nodes = list(nodes)
    # print(nodes)
    for node in nodes:
      # print("processing", node)
      path = __follow(node, doc_edge_index)
      # print("...",ok,0 in path, path)
      if 0 in path:
        continue
      doc_edge_index[0].append(node)
      doc_edge_index[1].append(0)
      doc_edge_weight.append(abs(node-0))
      ok = False
      break


def __text2syngraph(words):
  doc_edge_index = [[0],[0]]
  doc_edge_weight = [0]
  for index, word in enumerate(words): # text, lemma, pos, feats, dep, head, hurtlex
    index = index +1
    head = word["head"]
    head = int(head)
    #syntactic link
    if index == head: # parsing error
      doc_edge_index[0].append(index)
      doc_edge_index[1].append(0)
      doc_edge_weight.append(abs(index-0))
      # print(index,word["head"], word["text"], word["pos"], "[", index, 0, "]")
    else:
      doc_edge_index[0].append(index)
      doc_edge_index[1].append(head)
      doc_edge_weight.append(abs(index-head))
      # print(index,word["head"], word["text"], word["pos"], "[", index, head, "]")
  __add_discoannected_to_the_root(doc_edge_index, doc_edge_weight)
  #add self-loos
  for node in set(doc_edge_index[0]) | set(doc_edge_index[1]):
    doc_edge_index[0].append(node)
    doc_edge_index[1].append(node)
    doc_edge_weight.append(0)
  # print(doc_edge_index[0])
  # print(doc_edge_index[1])
  # print("."*80)
  return doc_edge_index, doc_edge_weight

def text2graph(words, connection_type, sliding_window=5):
  doc_edge_index = [[],[]]
  doc_edge_weight = []
  mem = set()
  for i, w in enumerate(words):
    w["position"] = i
  if connection_type == "dense":
    sliding_window = len(words)
  elif connection_type == "syntactic":
    return __text2syngraph(words)
  for context_window in ngrams(words, min(sliding_window,len(words))):
    # print(sliding_window, len(context_window), [words.index(v) for v in context_window])
    for word1 in context_window:
      w1 = words.index(word1)
      for word2 in context_window:
        w2 = words.index(word2) 
        str_pos = str(w1) + "_" + str(w2)
        if str_pos in mem:
          # print("avoiding double", str_pos)
          continue
        mem.add(str_pos)
        doc_edge_index[0].append(w1)
        doc_edge_index[1].append(w2)
        doc_edge_weight.append([w1-w2])
  # print(len(set(doc_edge_index[0])),len(set(doc_edge_index[1])), len(set(doc_edge_index[0]) | set(doc_edge_index[1])), len(words))
  # print([(i, w["text"]) for i, w in enumerate(words)])
  # print(set(doc_edge_index[0]) | set(doc_edge_index[1]))
  assert len(set(doc_edge_index[0]) | set(doc_edge_index[1]))  == len(words)
  return doc_edge_index, doc_edge_weight

def __profile_graph(edge_index, edge_type=None, plot_graph=False, words = None):
  G = nx.Graph()
  G1 = nx.Graph()
  for e1, e2 in zip(edge_index[0], edge_index[1]):
    G.add_edge(e1, e2) #, color='red', weight=0.84, size=300)
    if words:
      # print(len(words), e1, e2)
      e1 = words[e1]["text"]
      e2 = words[e2]["text"]
    G1.add_edge(e1, e2) #, color='red', weight=0.84, size=300)
  if plot_graph:
    if edge_type and edge_type == "syntactic":
      pos=nx.kamada_kawai_layout(G1)
    else:
      pos=nx.spiral_layout(G)
    nx.draw(G1,with_labels = True, pos=pos)
    plt.show()
  average_clustering_coefficient = nx.average_clustering(G)
  average_connectivity = nx.average_node_connectivity(G)
  avg_degree_connectivity = nx.average_degree_connectivity(G)
  
  acum = 0
  for cnt in avg_degree_connectivity:
    acum += cnt*avg_degree_connectivity[cnt]
  average_degree_connectivity = acum / sum(avg_degree_connectivity.keys())
  avg_neighbor_degree = nx.average_neighbor_degree(G)
  
  acum = 0
  for cnt in avg_neighbor_degree:
    acum += cnt*avg_neighbor_degree[cnt]
  average_neighbor_degree = acum / sum(avg_neighbor_degree.keys()) if sum(avg_neighbor_degree.keys())>0 else 0

  average_shortest_path_length = nx.average_shortest_path_length(G)
  diameter = nx.diameter(G)

  # print("average_clustering_coefficient",average_clustering_coefficient)
  # print("average_connectivity",average_connectivity)
  # print("average_degree_connectivity",average_degree_connectivity)
  # print("average_neighbor_degree", average_neighbor_degree)
  # print("average_shortest_path_length",average_shortest_path_length)
  # print("diameter",diameter)
  return {
      "average_clustering_coefficient": average_clustering_coefficient,
      "average_connectivity": average_connectivity,
      "average_degree_connectivity": average_degree_connectivity,
      "average_neighbor_degree": average_neighbor_degree,
      "average_shortest_path_length":average_shortest_path_length,
      "diameter":diameter
  }

def dataset2graph(ds, connection_type, sliding_window=5, profile_ds=True, doBERT=False):
  assert connection_type in ["dense", "ngram", "syntactic"]
  ds_profile = {}
  new_ds = []
  for index, doc in enumerate(tqdm(ds)):
    real_cls, words, lang = ds[doc]
    # print(doc, ds[doc])
    if doBERT:
      tk = tokenizer_de if lang=="de" else tokenizer_en if lang=="en" else tokenizer_es if lang=="es" else tokenizer_it if lang=="it" else None
      if connection_type != "syntactic":
        str_words = [w["text"] for w in words]
        new_tks = tk(" ".join(str_words))
        words = [{"text":w} for w in new_tks["input_ids"]]
      else:
        new_words = [{"text":"[CLS]", "lemma":"CLS", "pos":"None", "feats":"_", "dep":"None","head":0, "hurtlex": "NA"}]
        for w in words:
          # print(w)
          # input()
          new_tks = tk(w["text"])
          new_tks = new_tks["input_ids"]
          for new_tk in new_tks[1:-1]: # skip [cls] and [sep]
            new_tk = {"text":new_tk, "lemma":w["lemma"], "pos":w["pos"], "feats":w["feats"], "dep":w["dep"],"head":w["head"], "hurtlex": w["hurtlex"]}
            # print(len(words)+1, "\t",new_tk)
            new_words.append(new_tk)
        new_words.append({"text":"[SEP]", "lemma":"SEP", "pos":"None", "feats":"_", "dep":"None","head":0, "hurtlex": "NA"})
        words = new_words
        # input()
    edge_index, edge_weight = text2graph(words, connection_type, sliding_window=sliding_window)
    if profile_ds:
      words.insert(0,{"text":"root"})
      p = __profile_graph(edge_index, edge_type=connection_type, plot_graph=index<3, words= words)
      if real_cls not in ds_profile:
        ds_profile[real_cls] = {}
      for f in p:
        if f not in ds_profile[real_cls]:
          ds_profile[real_cls][f] = []
        ds_profile[real_cls][f].append(p[f])
    new_ds.append( {"id":doc, "edge_index":edge_index, "edge_weight":edge_weight, "cls":real_cls} )
    # if len(new_ds) > 1:
    #   break
  doc = new_ds[0]
  # print(doc["id"], "cls",doc["cls"])
  # print("...",doc["edge_index"])
  # print("-"*10)
  if profile_ds:
    for cls in ds_profile:
      for feat in ds_profile[cls]:
        lst = ds_profile[cls][feat]
        # lst = np.asarray(lst)
        print(cls,feat,np.mean(lst),np.std(lst),np.median(lst))
  return new_ds

def __load_cls_id_map(file_path):
  ds, _ = pickle.load(open(file_path, 'rb'))
  cls = {}
  for item in ds:
    cls[item["id"]] = (item["cls"], item["parsed"], item["lang"])
  print("cls len", len(cls))
  return cls


# BERT
# files = glob.glob(base_dir + "en_hasoc2021_train.csv.pkl")#HASOC 2021
files = glob.glob(base_dir + "only*.csv.pkl")
print(files)
edge_types = ["syntactic"]
for file_path in files:
  print(file_path)
  cls = __load_cls_id_map(file_path)
  for edge_type in edge_types:
    if edge_type == "ngram":
      for window in [3]:
        print("...window", window)
        if os.path.exists(file_path.replace(".pkl","") + "_" + edge_type +str(window) + "_subwords.pkl"):
          print(file_path.replace(".pkl","") + "_" + edge_type +str(window) + "_subwords.pkl", "alredy exists")
          continue
        new_ds = dataset2graph(ds=cls, connection_type=edge_type, sliding_window=window, doBERT=True)
        print("saving", file_path.replace(".pkl","") + "_" + edge_type+str(window)+".pkl")
        pickle.dump(new_ds, open(file_path.replace(".pkl","") + "_" + edge_type +str(window) + "_subwords.pkl", 'wb'))
    else:
      if os.path.exists(file_path.replace(".pkl","") + "_" + edge_type + "_subwords.pkl"):
          print(file_path.replace(".pkl","") + "_" + edge_type + "_subwords.pkl", "alredy exists")
          continue
      new_ds = dataset2graph(ds=cls, connection_type=edge_type, doBERT=True)
      print("saving", file_path.replace(".pkl","") + "_" + edge_type+".pkl")
      pickle.dump(new_ds, open(file_path.replace(".pkl","") + "_" + edge_type + "_subwords.pkl", 'wb'))

