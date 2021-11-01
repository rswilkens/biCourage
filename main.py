import argparse
import sys

from sklearn.model_selection import train_test_split

import models
import dataloader
import learner

parser = argparse.ArgumentParser(description='biCourage system!')

parser.add_argument('--model_name', required = True, type=str, dest='model_name', action='store', help='', choices=models.model_names) # model_name = sys.argv[1] #'0'
parser.add_argument('--experiment_name', required = True, type=str, dest='experiment_name', action='store', help='') #experiment_name = sys.argv[2] #'test'

parser.add_argument('--dataset_path', required = True, type=str, dest='dataset_path', action='store', help='') # dataset_path = sys.argv[3] # "../pkl/only_hateval.csv"
parser.add_argument('--encoding', required = True, type=str, dest='encoding', action='store', help='') # encoding  = sys.argv[6] # "wordembedding"
parser.add_argument('--edge_type', required = True, type=str, dest='edge_type', action='append', help='' , choices=['ngram3', 'syntactic', 'ngram3-subwords', 'syntactic-subwords']) # edge_type = sys.argv[4] # "ngram3"
parser.add_argument('--lang', required = True, type=str, dest='lang', action='store', help='', choices=['en', 'es', 'de', 'it']) # lang = sys.argv[5] # "en"
parser.add_argument('--lr', required = True, type=str, dest='lr', action='store', help='') # lr = sys.argv[7] # 5e-4

parser.add_argument('--actv', required = True, type=str, dest='actv', action='store', help='', choices=['selu', 'relu', 'elu']) # actv = sys.argv[9] # relu
parser.add_argument('--size', required = True, type=float, dest='size', action='store', help='') # size = float(sys.argv[8]) # 1
parser.add_argument('--num_layers', required = True, type=int, dest='num_layers', action='store', help='', choices=range(1,4)) # num_layers  = int(sys.argv[10]) # 2
parser.add_argument('--without_edge_weight', required = False, dest='use_edge_weight', action='store', help='') # use_edge_weight = sys.argv[11].lower() == "true"
parser.set_defaults(use_edge_weight=True)
parser.add_argument('--batch_size', required = False, type=float, default=-1, dest='batch_size', action='store', help='') # batch_size = float(sys.argv[12]) # -1
parser.add_argument('--run_nun_classes', required = False, type=int, default=2, dest='run_nun_classes', action='store', help='') # run_nun_classes = int(sys.argv[13])

parser.add_argument('--train_test_file', required = True, type=str, dest='train_test_file', action='store', help='') # train_test_file = sys.argv[14]
parser.add_argument('--test_ID', required = True, type=str, dest='test_ID', action='store', help='', choices=[str(i) for i in range(10)]) # test_ID = sys.argv[15]

args = parser.parse_args()

for key,value in vars(args).items():
    print(key, value)
# print("1 model_name",model_name)
# print("2 experiment_name",experiment_name)
# print("3 dataset_path",dataset_path)
# print("6 encoding",encoding)
# print("4 edge_type",edge_type)
# print("5 lang",lang)
# print("7 lr",lr)
# print("8 size",size)
# print("9 actv",actv)
# print("10 num_layers",num_layers)
# print("11 use_edge_weight",use_edge_weight)
# print("12 batch_size",batch_size)
# print("13 run_nun_classes",run_nun_classes)
# print("14 train_test_file",train_test_file)
# print("15 test_ID",test_ID)
# # input()

print("running", args.experiment_name, "at", args.model_name, "size",args.size, "lr",args.lr, "size",args.size, "num_layers", args.num_layers, "activation", args.actv, "batch_size", args.batch_size)
print("...dataset:",args.dataset_path, "encoding", args.encoding, "edge_type",args.edge_type, "language",args.lang)



graphs, maxlen = dataloader.load_graphs(dataset_path=args.dataset_path, encoding=args.encoding, edge_type=args.edge_type, lang=args.lang, nun_classes=args.run_nun_classes) 
# graphs = graphs[:500]


# run_nun_classes = 2
EPOCHS = 600
# model_name = "MeanPoolNetwork1"
experiment_name = args.experiment_name +"_"+ args.model_name + "_"+args.lr + "_" + args.test_ID

# all_train_data = graphs
# test_data = graphs

testset_ids = []

for ln in open(args.train_test_file).readlines():
    fold, id, cls = ln.strip().split("\t")
    if fold == args.test_ID:
        # print("DEBUG1: ",fold,test_ID, fold == test_ID)
        testset_ids.append(id)

#print("DEBUG2: ",testset_ids)

all_train_data,test_data = [],[]
all_labels = []

for g in graphs:
    if type(g) == list:
        gId = g[0].get_id
        all_labels.append(g[0].y[0])
    else:
        gId = g.get_id
        all_labels.append(g.y[0])
    # print(gId,gId in testset_ids, testset_ids[:10])
    if gId in testset_ids:
        test_data.append(g)
    else:
        all_train_data.append(g)

# all_train_data = all_train_data[:500]

num_classes = len(set(all_labels))
print(len(all_train_data+test_data), "datasets loaded", num_classes, "labels read:", 
    set(all_labels), 
    len(all_train_data), "instances for training and", len(test_data), "instances for testing")
assert num_classes > 1
assert num_classes == args.run_nun_classes

#
all_train_graphs = all_train_data 
test_data_graphs = test_data 
# test_data2, _ = dataloader.load_graphs(dataset_path=dataset_path.replace("train","test_task1"), encoding=encoding, edge_type=edge_type, lang=lang) 
# test_data_graphs2 = test_data2

stratify = []
for g in all_train_graphs:
    if type(g) == list:
        stratify.append(g[0].y[0])
    else:
        stratify.append(g.y[0])


train_graphs, val_graphs = train_test_split(all_train_graphs, test_size=0.1, stratify=stratify)


# all_train_data = test_data_graphs

if args.batch_size < 0:
    batch_size = len(train_graphs)
elif args.batch_size < 1:
    batch_size = int(len(train_graphs)*args.batch_size)
else:
    batch_size = int(args.batch_size)

model = models.get_model(num_classes = args.run_nun_classes, use_model=args.model_name, size = args.size, activation = args.actv, layers = args.num_layers, num_threads=len(args.edge_type))

model = learner.train(model=model, train_graphs=train_graphs, val_graphs=val_graphs, batch_size=batch_size, num_classes=args.run_nun_classes, 
       use_edge_weight=args.use_edge_weight, test_data_graphs=test_data_graphs,
       learning_rate=args.lr, epochs=EPOCHS, min_epochs=EPOCHS, experiment_name=args.experiment_name)

