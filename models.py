import learner

import numpy as np
import tensorflow as tf
import tf_geometric as tfg

import tensorflow_addons as tfa  # used for cocob

model_names = ["bicourageNonLinearInv"]

def get_model(num_classes= 2, use_model="bicourageNonLinear", size=0.5, activation="relu", layers=3, num_threads=1, network_params={}):
  assert use_model in model_names
  # if use_model == "bicouragesingleNonLinear":
  #   return bicouragesingleInvNonLinear(num_classes=num_classes,size=size, activation=activation, gcn_layers=layers ,drop_rate=network_params["drop_rate"] if "drop_rate" in network_params else 0.1)
  if use_model == "bicourageNonLinearInv":
    return bicourageInvNonLinear(num_classes=num_classes,size=size, activation=activation, gcn_layers=layers ,num_threads=num_threads,drop_rate=network_params["drop_rate"] if "drop_rate" in network_params else 0.1)
  # if use_model == "bicourageNonLinearInvDropout":
  #   return bicourageInvNonLinearDropout(num_classes=num_classes,size=size, activation=activation, gcn_layers=layers ,drop_rate=network_params["drop_rate"] if "drop_rate" in network_params else 0.1)



class bicourageInvNonLinear(tf.keras.Model):
    def __init__(self, num_classes, size,activation,gcn_layers, num_threads, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        # activation2 = tf.nn.softmax if "softmax" in activation else None
        if "selu" in activation: 
           activation = tf.nn.selu
        elif "relu" in activation:
           activation = tf.nn.relu
        elif "elu" in activation:
           activation = tf.nn.elu
        else:
           activation = None
        self.all_threads = []
        for _ in range(num_threads):
            thread = []
            sageSize = 100
            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
            gcn0 = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
            thread.append(gcn0)
            if gcn_layers>=2:
              sageSize = 150
              sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
              gcn1 = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
              thread.append(gcn1)
            if gcn_layers>=3:
              sageSize = 200
              sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
              gcn2 = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
              thread.append(gcn2)
            sageSize = 200
            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
            gcnF = tfg.layers.MeanGraphSage(sageSize, activation=activation)
            thread.append(gcnF)
            self.all_threads.append(thread)
        # sageSize = 100
        # sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        # self.gcn0A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
        # if gcn_layers>=2:
        #    sageSize = 150
        #    sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        #    self.gcn1A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
        # if gcn_layers>=3:
        #    sageSize = 200
        #    sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        #    self.gcn2A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
        # #
        # sageSize = 100
        # sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        # self.gcn0B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
        # if gcn_layers>=2:
        #    sageSize = 150
        #    sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        #    self.gcn1B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
        # if gcn_layers>=3:
        #    sageSize = 200
        #    sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        #    self.gcn2B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
        # sageSize = 200
        # sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
        # self.gcnFA = tfg.layers.MeanGraphSage(sageSize, activation=activation)
        # self.gcnFB = tfg.layers.MeanGraphSage(sageSize, activation=activation)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.dense1 = tf.keras.layers.Dense(50)
        self.dense2 = tf.keras.layers.Dense(10)
        self.dense3 = tf.keras.layers.Dense(num_classes)
        self.last_emb = None
        self.gcn_layers=gcn_layers
        #
    # @tf_utils.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        # assert len(inputs) == 2
        pools = []
        if len(inputs[0]) == 4:
            for (x, edge_index, edge_weight, node_graph_index), thread in zip(inputs, self.all_threads):
               h = thread[0]([x, edge_index, edge_weight])
               if self.gcn_layers>=2:
                  h = self.dropout(h, training=training)
                  h = thread[1]([h, edge_index, edge_weight])
               if self.gcn_layers>=3:
                  h = self.dropout(h, training=training)
                  h = thread[2]([h, edge_index, edge_weight])
               h = thread[-1]([h, edge_index, edge_weight])
               pool1 = tfg.nn.mean_pool(h, node_graph_index)
               pool2 = tfg.nn.min_pool(h, node_graph_index)
               pool3 = tfg.nn.max_pool(h, node_graph_index)
               pool4 = tfg.nn.sum_pool(h, node_graph_index)
               pools.extend([pool1, pool2, pool3, pool4])
        else:
            for (x, edge_index, node_graph_index), thread in zip(inputs, self.all_threads):
               h = thread[0]([x, edge_index])
               if self.gcn_layers>=2:
                  h = self.dropout(h, training=training)
                  h = thread[1]([h, edge_index])
               if self.gcn_layers>=3:
                  h = self.dropout(h, training=training)
                  h = thread[2]([h, edge_index])
               h = thread[-1]([h, edge_index])
               pool1 = tfg.nn.mean_pool(h, node_graph_index)
               pool2 = tfg.nn.min_pool(h, node_graph_index)
               pool3 = tfg.nn.max_pool(h, node_graph_index)
               pool4 = tfg.nn.sum_pool(h, node_graph_index)
               pools.extend([pool1, pool2, pool3, pool4])

        pool = tf.keras.layers.Concatenate()(pools) # ([poolA1, poolA2, poolA3, poolA4, poolB1, poolB2, poolB3, poolB4])

        self.last_emb = pool
        h = self.dropout(pool, training=training)
        # Predict Graph Labels
        h = self.dense1(h)
        h = self.dense2(h)
        h = self.dense3(h)
        #
        return h

# class bicourageInvNonLinear(tf.keras.Model):
#     def __init__(self, num_classes, size,activation,gcn_layers, drop_rate, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         #
#         # activation2 = tf.nn.softmax if "softmax" in activation else None
#         if "selu" in activation: 
#            activation = tf.nn.selu
#         elif "relu" in activation:
#            activation = tf.nn.relu
#         elif "elu" in activation:
#            activation = tf.nn.elu
#         else:
#            activation = None
#         sageSize = 100
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcn0A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=2:
#            sageSize = 150
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn1A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=3:
#            sageSize = 200
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn2A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         #
#         sageSize = 100
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcn0B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=2:
#            sageSize = 150
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn1B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=3:
#            sageSize = 200
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn2B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         sageSize = 200
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcnFA = tfg.layers.MeanGraphSage(sageSize, activation=activation)
#         self.gcnFB = tfg.layers.MeanGraphSage(sageSize, activation=activation)
#         self.dropout = tf.keras.layers.Dropout(drop_rate)
#         self.dense1 = tf.keras.layers.Dense(50)
#         self.dense2 = tf.keras.layers.Dense(10)
#         self.dense3 = tf.keras.layers.Dense(num_classes)
#         self.last_emb = None
#         self.gcn_layers=gcn_layers
#         #
#     # @tf_utils.function(experimental_relax_shapes=True)
#     def call(self, inputs, training=None, mask=None):
#         assert len(inputs) == 2
#         if len(inputs[0]) == 4:
#             [xA, edge_indexA, edge_weightA, node_graph_indexA], [xB, edge_indexB, edge_weightB, node_graph_indexB] = inputs
#             #print("gcnA0")
#             hA = self.gcn0A([xA, edge_indexA, edge_weightA])
#             if self.gcn_layers>=2:
#                hA = self.dropout(hA, training=training)
#                #print("gcnA1")
#                hA = self.gcn1A([hA, edge_indexA, edge_weightA])
#             if self.gcn_layers>=3:
#                hA = self.dropout(hA, training=training)
#                #print("gcnA2")
#                hA = self.gcn2A([hA, edge_indexA, edge_weightA])
#             hA = self.gcnFA([hA, edge_indexA, edge_weightA])
#             if edge_weightB.shape == ():
#               edge_weightB = np.asarray([0])
#             hB = self.gcn0B([xB, edge_indexB, edge_weightB])
#             if self.gcn_layers>=2:
#                hB = self.dropout(hB, training=training)
#                hB = self.gcn1B([hB, edge_indexB, edge_weightB])
#             if self.gcn_layers>=3:
#                hB = self.dropout(hB, training=training)  
#                hB = self.gcn2B([hB, edge_indexB, edge_weightB])
#             hB = self.gcnFA([hB, edge_indexB, edge_weightB])
#         else:
#             [xA, edge_indexA, node_graph_indexA], [xB, edge_indexB, node_graph_indexB] = inputs
#             edge_weight = None
#             hA = self.gcn0A([xA, edge_indexA])
#             if self.gcn_layers>=2:
#                hA = self.dropout(hA, training=training)
#                hA = self.gcn1A([hA, edge_indexA])
#             if self.gcn_layers>=3:
#                hA = self.dropout(hA, training=training)
#                hA = self.gcn2A([hA, edge_indexA])
#             hA = self.gcnFA([hA, edge_indexA])
#             hB = self.gcn0B([xB, edge_indexB])
#             if self.gcn_layers>=2:
#                hB = self.dropout(hB, training=training)
#                hB = self.gcn1B([hB, edge_indexB])
#             if self.gcn_layers>=3:
#                hB = self.dropout(hB, training=training)
#                hB = self.gcn2B([hB, edge_indexB])
#             hB = self.gcnFB([hB, edge_indexB])
#         # GCN Encoder
#         #
#         # Mean Pooling
#         poolA1 = tfg.nn.mean_pool(hA, node_graph_indexA)
#         poolA2 = tfg.nn.min_pool(hA, node_graph_indexA)
#         poolA3 = tfg.nn.max_pool(hA, node_graph_indexA)
#         poolA4 = tfg.nn.sum_pool(hA, node_graph_indexA)

#         poolB1 = tfg.nn.mean_pool(hB, node_graph_indexB)
#         poolB2 = tfg.nn.min_pool(hB, node_graph_indexB)
#         poolB3 = tfg.nn.max_pool(hB, node_graph_indexB)
#         poolB4 = tfg.nn.sum_pool(hB, node_graph_indexB)


#         pool = tf.keras.layers.Concatenate()([poolA1, poolA2, poolA3, poolA4, poolB1, poolB2, poolB3, poolB4])

#         self.last_emb = pool
#         h = self.dropout(pool, training=training)
#         #
#         # Predict Graph Labels
#         h = self.dense1(h)
#         h = self.dense2(h)
#         h = self.dense3(h)
#         #
#         return h

# class bicouragesingleInvNonLinear(tf.keras.Model):
#     def __init__(self, num_classes, size,activation,gcn_layers, drop_rate, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         #
#         # activation2 = tf.nn.softmax if "softmax" in activation else None
#         if "selu" in activation: 
#            activation = tf.nn.selu
#         elif "relu" in activation:
#            activation = tf.nn.relu
#         elif "elu" in activation:
#            activation = tf.nn.elu
#         else:
#            activation = None
#         netsize = 100
#         netsize = int(netsize*size) if int(netsize*size) % 2 == 0 else int(netsize*size)+1
#         self.gcn0A = tfg.layers.MeanGraphSage(netsize, activation=activation) 
#         if gcn_layers>=2:
#            netsize = 150
#            netsize = int(netsize*size)  if int(netsize*size) % 2 == 0 else int(netsize*size)+1
#            self.gcn1A = tfg.layers.MeanGraphSage(netsize, activation=activation) 
#         if gcn_layers>=3:
#            netsize = 200
#            netsize = int(netsize*size) if int(netsize*size) % 2 == 0 else int(netsize*size)+1
#            self.gcn2A = tfg.layers.MeanGraphSage(netsize, activation=activation) 
#         sageSize = 200
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcnFA = tfg.layers.MeanGraphSage(sageSize, activation=activation)
#         self.dropout = tf.keras.layers.Dropout(drop_rate)
#         self.dense1 = tf.keras.layers.Dense(50)
#         self.dense2 = tf.keras.layers.Dense(10)
#         self.dense3 = tf.keras.layers.Dense(num_classes)
#         self.last_emb = None
#         self.gcn_layers=gcn_layers
#         #
#     # @tf_utils.function(experimental_relax_shapes=True)
#     def call(self, inputs, training=None, mask=None):
#         if len(inputs) == 4:
#             [xA, edge_indexA, edge_weightA, node_graph_indexA] = inputs
#             #print("gcnA0")
#             hA = self.gcn0A([xA, edge_indexA, edge_weightA])
#             if self.gcn_layers>=2:
#                hA = self.dropout(hA, training=training)
#                #print("gcnA1")
#                hA = self.gcn1A([hA, edge_indexA, edge_weightA])
#             if self.gcn_layers>=3:
#                hA = self.dropout(hA, training=training)
#                #print("gcnA2")
#                hA = self.gcn2A([hA, edge_indexA, edge_weightA])
#             hA = self.gcnFA([hA, edge_indexA, edge_weightA])
#         else:
#             [xA, edge_indexA, node_graph_indexA]  = inputs
#             edge_weight = None
#             hA = self.gcn0A([xA, edge_indexA])
#             if self.gcn_layers>=2:
#                hA = self.dropout(hA, training=training)
#                hA = self.gcn1A([hA, edge_indexA])
#             if self.gcn_layers>=3:
#                hA = self.dropout(hA, training=training)
#                hA = self.gcn2A([hA, edge_indexA])
#             hA = self.gcnFA([hA, edge_indexA])
#         # GCN Encoder
#         #
#         # Mean Pooling
#         poolA1 = tfg.nn.mean_pool(hA, node_graph_indexA)
#         poolA2 = tfg.nn.min_pool(hA, node_graph_indexA)
#         poolA3 = tfg.nn.max_pool(hA, node_graph_indexA)
#         poolA4 = tfg.nn.sum_pool(hA, node_graph_indexA)

#         pool = tf.keras.layers.Concatenate()([poolA1, poolA2, poolA3, poolA4])

#         self.last_emb = pool
#         h = self.dropout(pool, training=training)
#         #
#         # Predict Graph Labels
#         h = self.dense1(h)
#         h = self.dense2(h)
#         h = self.dense3(h)
#         #
#         return h



# class bicourageInvNonLinearDropout(tf.keras.Model):
#     def __init__(self, num_classes, size,activation,gcn_layers, drop_rate, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         #
#         # activation2 = tf.nn.softmax if "softmax" in activation else None
#         if "selu" in activation: 
#            activation = tf.nn.selu
#         elif "relu" in activation:
#            activation = tf.nn.relu
#         elif "elu" in activation:
#            activation = tf.nn.elu
#         else:
#            activation = None
#         sageSize = 100
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcn0A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=2:
#            sageSize = 150
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn1A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=3:
#            sageSize = 200
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn2A = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         #
#         sageSize = 100
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcn0B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=2:
#            sageSize = 150
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn1B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         if gcn_layers>=3:
#            sageSize = 200
#            sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#            self.gcn2B = tfg.layers.MeanGraphSage(sageSize, activation=activation) 
#         sageSize = 200
#         sageSize = int(sageSize*size) if int(sageSize*size) % 2 == 0 else int(sageSize*size)+1
#         self.gcnFA = tfg.layers.MeanGraphSage(sageSize, activation=activation)
#         self.gcnFB = tfg.layers.MeanGraphSage(sageSize, activation=activation)
#         self.dropout = tf.keras.layers.Dropout(drop_rate)
#         self.dense1 = tf.keras.layers.Dense(50)
#         self.dense2 = tf.keras.layers.Dense(10)
#         self.dense3 = tf.keras.layers.Dense(num_classes)
#         self.last_emb = None
#         self.gcn_layers=gcn_layers
#         #
#     # @tf_utils.function(experimental_relax_shapes=True)
#     def call(self, inputs, training=None, mask=None):
#         assert len(inputs) == 2
#         #print("==>", len(inputs), len(inputs[0]), len(inputs[1]) )
#         if len(inputs[0]) == 4:
#             [xA, edge_indexA, edge_weightA, node_graph_indexA], [xB, edge_indexB, edge_weightB, node_graph_indexB] = inputs
#             #print("gcnA0")
#             hA = self.gcn0A([xA, edge_indexA, edge_weightA])
#             if self.gcn_layers>=2:
#                hA = self.dropout(hA, training=training)
#                #print("gcnA1")
#                hA = self.gcn1A([hA, edge_indexA, edge_weightA])
#             if self.gcn_layers>=3:
#                hA = self.dropout(hA, training=training)
#                #print("gcnA2")
#                hA = self.gcn2A([hA, edge_indexA, edge_weightA])
#             hA = self.dropout(hA, training=training)
#             hA = self.gcnFA([hA, edge_indexA, edge_weightA])
#             # print("\t\t",xB.shape, edge_indexB.shape, edge_weightB.shape)
#             # print("\t--\t",xB, edge_indexB, edge_weightB)
#             if edge_weightB.shape == ():
#               edge_weightB = np.asarray([0])
#             hB = self.gcn0B([xB, edge_indexB, edge_weightB])
#             if self.gcn_layers>=2:
#                hB = self.dropout(hB, training=training)
#                #print("gcnB1")
#                hB = self.gcn1B([hB, edge_indexB, edge_weightB])
#             if self.gcn_layers>=3:
#                hB = self.dropout(hB, training=training)  
#                #print("gcnB2")
#                hB = self.gcn2B([hB, edge_indexB, edge_weightB])
#             hB = self.dropout(hB, training=training)  
#             hB = self.gcnFA([hB, edge_indexB, edge_weightB])
#         else:
#             [xA, edge_indexA, node_graph_indexA], [xB, edge_indexB, node_graph_indexB] = inputs
#             edge_weight = None
#             hA = self.gcn0A([xA, edge_indexA])
#             if self.gcn_layers>=2:
#                hA = self.dropout(hA, training=training)
#                hA = self.gcn1A([hA, edge_indexA])
#             if self.gcn_layers>=3:
#                hA = self.dropout(hA, training=training)
#                hA = self.gcn2A([hA, edge_indexA])
#             hA = self.dropout(hA, training=training)
#             hA = self.gcnFA([hA, edge_indexA])
#             hB = self.gcn0B([xB, edge_indexB])
#             if self.gcn_layers>=2:
#                hB = self.dropout(hB, training=training)
#                hB = self.gcn1B([hB, edge_indexB])
#             if self.gcn_layers>=3:
#                hB = self.dropout(hB, training=training)
#                hB = self.gcn2B([hB, edge_indexB])
#             hB = self.dropout(hB, training=training)
#             hB = self.gcnFB([hB, edge_indexB])
#         # GCN Encoder
#         #
#         # Mean Pooling
#         poolA1 = tfg.nn.mean_pool(hA, node_graph_indexA)
#         poolA2 = tfg.nn.min_pool(hA, node_graph_indexA)
#         poolA3 = tfg.nn.max_pool(hA, node_graph_indexA)
#         poolA4 = tfg.nn.sum_pool(hA, node_graph_indexA)

#         poolB1 = tfg.nn.mean_pool(hB, node_graph_indexB)
#         poolB2 = tfg.nn.min_pool(hB, node_graph_indexB)
#         poolB3 = tfg.nn.max_pool(hB, node_graph_indexB)
#         poolB4 = tfg.nn.sum_pool(hB, node_graph_indexB)

#         pool = tf.keras.layers.Concatenate()([poolA1, poolA2, poolA3, poolA4, poolB1, poolB2, poolB3, poolB4])

#         self.last_emb = pool
#         h = self.dropout(pool, training=training)
#         #
#         # Predict Graph Labels
#         h = self.dense1(h)
#         h = self.dense2(h)
#         h = self.dense3(h)
#         #
#         return h