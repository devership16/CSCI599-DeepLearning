from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

####
# The following code uses artificially generated and very small networks.
# Expect less than excellent performance as these random networks do not have any interesting structure.
# The purpose of main.py is to show how to use the code!
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Replace dummy toy datasets used here with the actual datasets you just downloaded.
# (3) Train & test the model.
####

val_test_size = 0.05
data_folder = "./"

nodes_map = defaultdict(set)
with open(data_folder + "csvNodes20000_20.pickle", "rb") as _orgf:
    for _data in tqdm(pickle.load(_orgf)):
        if _data:
            _ds = _data.strip().split("_")
            nodes_map[_ds[0]].add('_'.join(_ds))

# Map for Organization ids
relationships = {}
with open(data_folder + "csvRelationships20000_20.pickle", "rb") as _orgf:
    for _data in tqdm(pickle.load(_orgf)):
        if _data:
            _ds = _data.strip().split("_")
            _key = (_ds[0], _ds[2])
            if _key not in relationships: relationships[_key] = defaultdict(list)
            _edge, _n1, _n2 = '_'.join(_ds[4:]), '_'.join(_ds[:2]), '_'.join(_ds[2:4])
            
            if _n1 not in nodes_map[_ds[0]]: nodes_map[_ds[1]].add(_n1)
            if _n2 not in nodes_map[_ds[2]]: nodes_map[_ds[3]].add(_n2)
            relationships[_key][_edge].append((_n1, _n2))

companies, CCI = set(), relationships[('Organization', 'Organization')]

for _e in CCI:
    for _c1, _c2 in CCI[_e]:
        companies.add(_c1)
        companies.add(_c2)

persons, CPI = set(), relationships[('Organization', 'Person')]

for _e in CPI:
    for _c, _p in CPI[_e]:
        if _c not in companies: companies.add(_c)
        persons.add(_p)

n_persons = len(persons)
map_persons = {_p:_i for _i, _p in enumerate(list(persons))}

brps, CBI = set(), relationships[('Organization', 'Bankruptcy')]

for _e in CBI:
    for _c, _i in CBI[_e]:
        if _c not in companies: companies.add(_c)
        brps.add(_i)

n_brps = len(brps)
map_brps = {_p:_i for _i, _p in enumerate(list(brps))}

n_companies = len(companies)  
n_compcomp_rel_types = len(relationships[('Organization', 'Organization')])
map_companies = {_p:_i for _i, _p in enumerate(list(companies))}
print(n_companies, n_persons, n_brps)

comp_comp_adj_list = []
edges_CCI = []
for _e in tqdm(CCI):
    if len(CCI[_e]) > 228:
        _mat = np.array([[map_companies[_u], map_companies[_v]] for _u, _v in CCI[_e]])
        _data = np.ones(len(CCI[_e]))
        comp_comp_adj_list.append(sp.csr_matrix((_data, (_mat[:, 0], _mat[:, 1])), shape=(n_companies, n_companies)))
        edges_CCI.append(_e)
comp_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in comp_comp_adj_list]

pers_comp_adj_list = []
edges_CPI = []
for _e in tqdm(CPI):
    if len(CPI[_e]) > 228:
        _mat = np.array([[map_persons[_v], map_companies[_u]] for _u, _v in CPI[_e]])    
        _data = np.ones(len(CPI[_e]))
        pers_comp_adj_list.append(sp.csr_matrix((_data, (_mat[:, 0], _mat[:, 1])), shape=(n_persons, n_companies)))
        edges_CPI.append(_e)
pers_comp_degrees_list = [np.array(_adj.sum(axis=0)).squeeze() for _adj in pers_comp_adj_list]

comp_bankr_adj = []

for _e in tqdm(CBI):
    if len(CBI[_e]) > 164:
        _mat = np.array([[map_companies[_u], map_brps[_v]] for _u, _v in CBI[_e]])    
        _data = np.ones(len(CBI[_e]))
        comp_bankr_adj.append(sp.csr_matrix((_data, (_mat[:, 0], _mat[:, 1])), shape=(n_companies, 1)))

comp_bankr_deg = [np.array(_adj.sum(axis=0)).squeeze() for _adj in comp_bankr_adj]


# data representation
adj_mats_orig = {
#     (0, 0): [pers_adj, pers_adj.transpose(copy=True)],
    (0, 1): pers_comp_adj_list,
    (1, 0): [x.transpose(copy=True) for x in pers_comp_adj_list],
    (1, 1): comp_comp_adj_list + [x.transpose(copy=True) for x in comp_comp_adj_list],
    (1, 2): comp_bankr_adj,
    (2, 1): [x.transpose(copy=True) for x in comp_bankr_adj]
}
degrees = {
    0: pers_comp_degrees_list,
    1: comp_degrees_list + comp_degrees_list,
    2: [np.array([np.sum(comp_bankr_adj[0].T)]), np.array([np.sum(comp_bankr_adj[0].T)])]
}


# features (Person)
pers_feat = sp.identity(n_persons)
pers_nonzero_feat, pers_num_feat = pers_feat.shape
pers_feat = preprocessing.sparse_to_tuple(pers_feat.tocoo())

# features (Companies)
comp_feat = sp.identity(n_companies)
comp_nonzero_feat, comp_num_feat = comp_feat.shape
comp_feat = preprocessing.sparse_to_tuple(comp_feat.tocoo())

# features (Bankruptcy)
n_bankruptcy = 1
banrp_feat = sp.identity(n_bankruptcy)
banrp_nonzero_feat, banrp_num_feat = banrp_feat.shape
banrp_feat = preprocessing.sparse_to_tuple(banrp_feat.tocoo())

# data representation
num_feat = {
    0: pers_num_feat,
    1: comp_num_feat,
    2: banrp_num_feat
}
nonzero_feat = {
    0: pers_nonzero_feat,
    1: comp_nonzero_feat,
    2: banrp_nonzero_feat
}
feat = {
    0: pers_feat,
    1: comp_feat,
    2: banrp_feat
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 1): 'dedicom',
    (1, 0): 'dedicom',
    (1, 1): 'dedicom',
    (1, 2): 'bilinear',
    (2, 1): 'bilinear'
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 64, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)
tf.app.flags.DEFINE_string('f', '', 'kernel')

# Create minibatch iterator

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

# create model

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

# create optimizer

print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

print("Initialize session")
sess = tf.Session()
# FileWriter("output", sess.graph)
sess.run(tf.global_variables_initializer())
feed_dict = {}


###########################################################
#
# Train model
#
###########################################################

###########################################################
#
# Train model
#
###########################################################

print("Train model")
for epoch in range(20):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()
        
        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()

