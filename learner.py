import os
from sklearn.metrics import classification_report
from tqdm import tqdm
import tensorflow as tf
import tf_geometric as tfg
import tensorflow_addons as tfa  # used for cocob
import numpy as np

IGNORE_GRAPH = False
DUMMY_BATCH = False

def detailed_score(real, pred):
  report = classification_report(real, pred, digits=3, output_dict=True, zero_division=0)
  detailed_score_val = {}
  for item in report:
    if item == "accuracy":
      detailed_score_val["accuracy"] = report["accuracy"]
    else:
      for score in report[item]:
        detailed_score_val[item + "_" + score] = report[item][score]
  return detailed_score_val

def save_epoch_log(output_log, current_epoch, min_epoch, scoresTrain, scoresVal, mean_loss_train, mean_loss_val, output_sequence=[], sep="\t"):
  if output_sequence==None or len(output_sequence)==0:
    output_sequence.extend(list(set(scoresTrain.keys()) | set(scoresVal.keys())))
    ln = "current_epoch" +sep+ "min_epoch" +sep+ "mean_loss_train" +sep+ "mean_loss_val" 
    lnTrain = ["train_" + score for score in output_sequence]
    lnVal = ["val_" + score for score in output_sequence]
    output_log.write(ln +sep+ sep.join(lnTrain) +sep+ sep.join(lnVal) + "\n") 
  # 
  ln = str(current_epoch) +sep+ str(min_epoch) +sep+ str(mean_loss_train) +sep+ str(mean_loss_val) 
  lnTrain = [str(scoresTrain[score]) if score in scoresTrain else "-" for score in output_sequence]
  lnVal = [str(scoresVal[score]) if score in scoresVal else "-" for score in output_sequence]
  output_log.write(ln +sep+ sep.join(lnTrain) +sep+ sep.join(lnVal) + "\n")

def create_graph_generator(graphs, batch_size, shuffle=False):
    # if IGNORE_GRAPH: # CNN
    #     datasetX = []
    #     datasetY = []
    #     for x,y in graphs:
    #         datasetX.append(x)
    #         datasetY.append(y)
    #     dataset = tf.data.Dataset.from_tensor_slices((datasetX, datasetY))
    #     if shuffle:
    #         dataset.shuffle(2000)
    #     dataset = dataset.batch(batch_size)
    #     return dataset.as_numpy_iterator()

    # else: # GCN
    dataset = tf.data.Dataset.range(len(graphs))
    if shuffle:
        dataset = dataset.shuffle(2000)
    dataset = dataset.batch(batch_size)
    #
    # print(dataset)
    for batch_graph_index in dataset:
      # if type(graphs[0]) == list:
        batch_graph_list = {}
        for i in batch_graph_index:
          for index, g in enumerate(graphs[i]):
            # print(i, index, g)
            if index not in batch_graph_list:
              batch_graph_list[index] = []
            batch_graph_list[index].append(g)
        instance = []
        # print("batch_graph_list", len(batch_graph_list), batch_graph_list)
        # input("press")
        for i in batch_graph_list:
          batch_graph = tfg.BatchGraph.from_graphs(batch_graph_list[i])
          instance.append(batch_graph)
        yield instance
        # batch_graph_list1 = [graphs[i][0] for i in batch_graph_index]
        # batch_graph_list2 = [graphs[i][1] for i in batch_graph_index]

        # batch_graph1 = tfg.BatchGraph.from_graphs(batch_graph_list1)
        # batch_graph2 = tfg.BatchGraph.from_graphs(batch_graph_list2)
        # yield [batch_graph1,batch_graph2]
      # else:
      #   batch_graph_list = [graphs[i] for i in batch_graph_index]
      #   batch_graph = tfg.BatchGraph.from_graphs(batch_graph_list)
        # yield batch_graph

def forward(model, batch_graph,use_edge_weight, training=None):
    # print("use_edge_weight",use_edge_weight)
    # print("batch_graph", len(batch_graph), type(batch_graph))
    # print("batch_graph[0]", type(batch_graph[0]))
    inputs = []
    for g in batch_graph:
      if use_edge_weight:
        inputs.append([g.x, g.edge_index, np.squeeze(g.edge_weight), g.node_graph_index])
      else:
        inputs.append([g.x, g.edge_index, g.node_graph_index])
    
    # print(len(inputs),len(inputs[0]))
    assert len(inputs) >= 1 and (len(inputs[0])==3 or len(inputs[0])==4)
    return model(inputs, training=training)

# def forward(model, batch_graph,use_edge_weight, training=None):
#     if IGNORE_GRAPH:
#         return model(batch_graph, training=training)
#     else:
#         if use_edge_weight:
#           if type(batch_graph) == list:
#             edge_weight0 = np.squeeze(batch_graph[0].edge_weight)
#             edge_weight1 = np.squeeze(batch_graph[1].edge_weight)
#             inputs = [[batch_graph[0].x, batch_graph[0].edge_index, edge_weight0, batch_graph[0].node_graph_index], [batch_graph[1].x, batch_graph[1].edge_index, edge_weight1, batch_graph[1].node_graph_index]]
#           else:
#             edge_weight = np.squeeze(batch_graph.edge_weight)
#             inputs = [batch_graph.x, batch_graph.edge_index, edge_weight, batch_graph.node_graph_index]
#         else:
#           if type(batch_graph) == list:
#             inputs = [[batch_graph[0].x, batch_graph[0].edge_index, batch_graph[0].node_graph_index], [batch_graph[0].x, batch_graph[0].edge_index, batch_graph[0].node_graph_index]]
#           else:
#             inputs = [batch_graph.x, batch_graph.edge_index, batch_graph.node_graph_index]
#         assert len(inputs)==3 or len(inputs)==4 or (len(inputs) == 2 and ((len(inputs[0])==3 and len(inputs[1])==3) or (len(inputs[0])==4 and len(inputs[1])==4)))
#         return model(inputs, training=training)

def compute_loss(logits, labels, num_classes):
      losses = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits,
          labels=tf.one_hot(labels, depth=num_classes)
      )
      mean_loss = tf.reduce_mean(losses)
      return mean_loss



def __store_pred(graphs,batch_size,model, num_classes, use_edge_weight, experiment_name, epoch, calc_report, predType="Validation"):
    if DUMMY_BATCH:
        mean_loss, all_preds, all_reals, all_logits, all_last_emb = __calc_dummy_batch(graphs=graphs, batch_size=batch_size, optimizer=None, model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, training=False, shuffle=False)
    else: 
        mean_loss, all_preds, all_reals, all_logits, all_last_emb = __calc_batch(graphs=graphs, batch_size=batch_size, optimizer=None, model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, training=False, shuffle=False)

    if calc_report:
        print('\n\n')
        classification_report_str = classification_report(all_reals, all_preds, digits=4, output_dict=False, zero_division=0)
        print(classification_report_str)

    assert len(all_preds) == len(all_logits)
    assert len(all_preds) == len(all_last_emb)
    assert len(all_preds) == len(graphs)
    with open(experiment_name + "/" + str(epoch) + ".pred" + predType, "w") as output_file:
      for p, l, e, g in zip(all_preds, all_logits, all_last_emb, graphs):
        if type(g) == list:
          g = g[0]
        selection = [str(v) for v in l]
        output_file.write(g.get_id + "," + str(p)+";" + ",".join(selection) +";" +",".join([str(v) for v in e]) + "\n")

def store_preds(val_graphs, test_graphs, train_graphs, use_edge_weight, model, experiment_name, epoch, num_classes, calc_report=False, batch_size = None):
    if batch_size == None:
      batch_size1=len(val_graphs)
      batch_size2=len(test_graphs)
      batch_size3=len(train_graphs)
    else:
      batch_size1=batch_size
      batch_size2=batch_size
      batch_size3=batch_size
    
    __store_pred(graphs=val_graphs,
                 batch_size=batch_size1,
                 model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, experiment_name=experiment_name, epoch=epoch, calc_report=calc_report,
                 predType="Validation")

    __store_pred(graphs=test_graphs,
                 batch_size=batch_size2,
                 model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, experiment_name=experiment_name, epoch=epoch, calc_report=calc_report,
                 predType="TestSet")

    __store_pred(graphs=train_graphs,
                 batch_size=batch_size3,
                 model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, experiment_name=experiment_name, epoch=epoch, calc_report=calc_report,
                 predType="Trainset")


def __calc_dummy_batch(graphs, batch_size, optimizer, model, num_classes, use_edge_weight, training=False, shuffle=True):
  all_last_emb = []
  all_logits = []
  all_preds = []
  all_reals = []

  steps = int(len(graphs)/batch_size)
  epoch_loss = []
  train_batch_generator = create_graph_generator(graphs, 1, shuffle=shuffle)
  for batch in range(steps):
      batch_reals = []
      batch_logits = []
      with tf.GradientTape() as tape:
        for batch_size_index in range(batch_size):
          train_batch_graph = next(train_batch_generator)
          one_logits = forward(model,train_batch_graph, use_edge_weight=use_edge_weight, training=training) 
          if type(train_batch_graph)==list:
            one_real = train_batch_graph[0].y
          else:
            one_real = train_batch_graph.y
          batch_logits.append(one_logits)
          all_logits.extend(one_logits.numpy().tolist())
          all_last_emb.extend(model.last_emb.numpy().tolist())
          batch_reals.append(one_real)
          all_reals.append(one_real)
          one_pred = tf.argmax(one_logits, axis=-1)
          all_preds.append(one_pred)
        mean_loss = compute_loss(batch_logits, batch_reals, num_classes)
      epoch_loss.append(mean_loss.numpy())
      if training:
        vars = tape.watched_variables()
        grads = tape.gradient(mean_loss, vars) 
        optimizer.apply_gradients(zip(grads, vars))
  if training: 
    return epoch_loss, all_preds, all_reals
  else:
    return epoch_loss, all_preds, all_reals, all_logits, all_last_emb

def __calc_batch(graphs, batch_size, optimizer, model, num_classes, use_edge_weight, training=False, shuffle=True):
  all_last_emb = []
  all_logits = []
  all_preds = []
  all_reals = []

  steps = int(len(graphs)/batch_size)
  epoch_loss = []
  train_batch_generator = create_graph_generator(graphs, batch_size, shuffle=shuffle)
  for batch in range(steps):
    train_batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        logits = forward(model,train_batch_graph, use_edge_weight=use_edge_weight, training=training) 
        if type(train_batch_graph)==list:
          real = train_batch_graph[0].y
        else:
          real = train_batch_graph.y
        mean_loss = compute_loss(logits, real, num_classes)
    if not training: # predict mode
      all_logits.extend(logits.numpy().tolist())
      all_last_emb.extend(model.last_emb.numpy().tolist())
    epoch_loss.append(mean_loss.numpy())
    preds = tf.argmax(logits, axis=-1)
    all_preds.extend(preds)
    all_reals.extend(real)
    if training:
      vars = tape.watched_variables()
      grads = tape.gradient(mean_loss, vars)
      optimizer.apply_gradients(zip(grads, vars))
    #  
  if training: 
    return epoch_loss, all_preds, all_reals
  else:
    return epoch_loss, all_preds, all_reals, all_logits, all_last_emb

def train(model, train_graphs, val_graphs, batch_size, num_classes, use_edge_weight, test_data_graphs,learning_rate=5e-5, epochs=100, verbose = True, early_stop_patience = 2, min_epochs = 40, experiment_name="tmp"):
  # 
  min_loss = 100000
  min_epoch = -1
  if "cocob" in learning_rate:
    optimizer = tfa.optimizers.COCOB() # https://arxiv.org/pdf/1705.07795.pdf
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))

  output_sequence = []
  if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
  checkpoint_prefix = os.path.join(experiment_name, "ckpt")
  if not os.path.exists(checkpoint_prefix):
    os.makedirs(checkpoint_prefix)
  with open(os.path.join(experiment_name, "log.csv"),"w") as output_log:
    steps = int(len(train_graphs)/batch_size)
    pbar = tqdm(total = epochs*steps)
    for epoch in range(epochs): # mini-batch learning
      if DUMMY_BATCH:
        mean_loss, all_preds, all_reals = __calc_dummy_batch(graphs=train_graphs, batch_size=batch_size, optimizer=optimizer, model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, training=True)
      else: 
        mean_loss, all_preds, all_reals = __calc_batch(graphs=train_graphs, batch_size=batch_size, optimizer=optimizer, model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, training=True)

      mean_loss_train = sum(mean_loss)/len(mean_loss)
      scoresTrain = detailed_score(all_reals, all_preds)
      #

      if DUMMY_BATCH:
        mean_loss, all_preds, all_reals, _, _ = __calc_dummy_batch(graphs=val_graphs, batch_size=len(val_graphs), optimizer=optimizer, model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, training=False, shuffle=False)
      else: 
        mean_loss, all_preds, all_reals, _, _ = __calc_batch(graphs=val_graphs, batch_size=len(val_graphs), optimizer=optimizer, model=model, num_classes=num_classes, use_edge_weight=use_edge_weight, training=False, shuffle=False)

      corrects = tf.cast(tf.equal(all_preds, all_reals), tf.float32)
      accuracy = tf.reduce_mean(corrects).numpy()
      scoresVal = detailed_score(all_reals, all_preds)
      mean_loss = sum(mean_loss)/len(mean_loss)
      if min_loss >mean_loss:
            min_loss = mean_loss
            min_epoch = epoch
      # 
      save_epoch_log(output_log=output_log, current_epoch=epoch, min_epoch=min_epoch, scoresTrain=scoresTrain, scoresVal=scoresVal, 
        mean_loss_train=mean_loss_train, mean_loss_val=mean_loss, output_sequence=output_sequence) 
      ckpt_prefix = os.path.join(checkpoint_prefix, "weights_" + str(epoch))
      store_preds(val_graphs=val_graphs, test_graphs=test_data_graphs, train_graphs=train_graphs,
                use_edge_weight=use_edge_weight, 
                model=model, 
                experiment_name=experiment_name, 
                epoch=epoch, batch_size = None, num_classes=num_classes)
      model.save_weights(ckpt_prefix) # checkpoint.save(file_prefix=checkpoint_prefix)
      if verbose: 
          pbar.set_description("train loss %s; validation loss %s; val accuracy %s; es %s" % (mean_loss_train, mean_loss, accuracy, epoch-min_epoch-early_stop_patience))
          pbar.update()
      if epoch-min_epoch-early_stop_patience >=0 and epoch > min_epochs:
          print("early_stop_patience", early_stop_patience, "best epoch", min_epoch, "current train loss", mean_loss_train,
                "current validation loss", mean_loss, "best loss", min_loss, "(min_epochs: ", min_epochs, ")")
          pbar.update()
          break
          # 
    pbar.close()
  return model

