from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from optparse import OptionParser
import numpy as np
import tensorflow as tf
from scipy.sparse import load_npz
from itertools import izip
import json


parser = OptionParser()
parser.add_option("-d", "--d", default=64, help="The embedding dimension d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=32, help="The dimension of project matrices k")
parser.add_option("-t", "--t", default="o", help="Test scenario")
(opts, args) = parser.parse_args()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


# load network
network_path = '../../../Databases/HMDAD/'

disease_semantic = np.loadtxt(network_path + 'disease_disease_HMDAD_semantic.txt')
disease_symptom = np.loadtxt(network_path + 'disease_disease_HMDAD_symptom.txt')
microbe_taxa_order = np.loadtxt(network_path + 'microbe_microbe_taxonomy_order_HMDAD.txt')

# normalize network for mean pooling aggregation
disease_semantic_normalize = row_normalize(disease_semantic, True)
disease_symptom_normalize = row_normalize(disease_symptom, True)
microbe_taxa_order_normalize = row_normalize(microbe_taxa_order, True)

# define computation graph
num_disease = len(disease_semantic_normalize)
num_microbe = len(microbe_taxa_order_normalize)
print "number of diseases:", num_disease
print "number of microbes:", num_microbe

dim_disease = int(opts.d)
dim_microbe = int(opts.d)
dim_pass = int(opts.d)
dim_pred = int(opts.k)


class Model(object):
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        # inputs
        self.disease_semantic = tf.placeholder(tf.float32, [num_disease, num_disease])
        self.disease_semantic_normalize = tf.placeholder(tf.float32, [num_disease, num_disease])

        self.disease_symptom = tf.placeholder(tf.float32, [num_disease, num_disease])
        self.disease_symptom_normalize = tf.placeholder(tf.float32, [num_disease, num_disease])

        self.disease_interact = tf.placeholder(tf.float32, [num_disease, num_disease])
        self.disease_interact_normalize = tf.placeholder(tf.float32, [num_disease, num_disease])

        self.microbe_taxa = tf.placeholder(tf.float32, [num_microbe, num_microbe])
        self.microbe_taxa_normalize = tf.placeholder(tf.float32, [num_microbe, num_microbe])

        self.microbe_interact = tf.placeholder(tf.float32, [num_microbe, num_microbe])
        self.microbe_interact_normalize = tf.placeholder(tf.float32, [num_microbe, num_microbe])

        self.disease_microbe_elevated = tf.placeholder(tf.float32, [num_disease, num_microbe])
        self.disease_microbe_elevated_normalize = tf.placeholder(tf.float32, [num_disease, num_microbe])

        self.disease_microbe_reduced = tf.placeholder(tf.float32, [num_disease, num_microbe])
        self.disease_microbe_reduced_normalize = tf.placeholder(tf.float32, [num_disease, num_microbe])

        self.microbe_disease_elevated = tf.placeholder(tf.float32, [num_microbe, num_disease])
        self.microbe_disease_elevated_normalize = tf.placeholder(tf.float32, [num_microbe, num_disease])

        self.microbe_disease_reduced = tf.placeholder(tf.float32, [num_microbe, num_disease])
        self.microbe_disease_reduced_normalize = tf.placeholder(tf.float32, [num_microbe, num_disease])

        self.disease_microbe_elevated_mask = tf.placeholder(tf.float32, [num_disease, num_microbe])
        self.disease_microbe_reduced_mask = tf.placeholder(tf.float32, [num_disease, num_microbe])

        # features
        self.disease_embedding = weight_variable([num_disease, dim_disease])
        self.microbe_embedding = weight_variable([num_microbe, dim_microbe])

        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.disease_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.microbe_embedding))

        # feature passing weights (different types of nodes can use different weights)
        W0 = weight_variable([dim_pass + dim_disease, dim_pass])
        b0 = bias_variable([dim_pass])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        # passing 1 times (can be easily extended to multiple passes)
        disease_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.disease_semantic_normalize, a_layer(self.disease_embedding, dim_pass)) +
                       tf.matmul(self.disease_symptom_normalize, a_layer(self.disease_embedding, dim_pass)) +
                       tf.matmul(self.disease_interact_normalize, a_layer(self.disease_embedding, dim_pass)) +
                       tf.matmul(self.disease_microbe_elevated_normalize, a_layer(self.microbe_embedding, dim_pass)) +
                       tf.matmul(self.disease_microbe_reduced_normalize, a_layer(self.microbe_embedding, dim_pass)),
                       self.disease_embedding], axis=1), W0)+b0), dim=1)

        microbe_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.microbe_taxa_normalize, a_layer(self.microbe_embedding, dim_pass)) +
                       tf.matmul(self.microbe_interact_normalize, a_layer(self.microbe_embedding, dim_pass)) +
                       tf.matmul(self.microbe_disease_elevated_normalize, a_layer(self.disease_embedding, dim_pass)) +
                       tf.matmul(self.microbe_disease_reduced_normalize, a_layer(self.disease_embedding, dim_pass)),
                       self.microbe_embedding], axis=1), W0)+b0), dim=1)

        self.disease_representation = disease_vector1
        self.microbe_representation = microbe_vector1

        # reconstructing networks
        self.disease_semantic_reconstruct = bi_layer(self.disease_representation,
                                                     self.disease_representation,
                                                     sym=True,
                                                     dim_pred=dim_pred)
        self.disease_semantic_reconstruct_loss = tf.reduce_sum(tf.multiply(
            (self.disease_semantic_reconstruct - self.disease_semantic),
            (self.disease_semantic_reconstruct - self.disease_semantic)))

        self.disease_symptom_reconstruct = bi_layer(self.disease_representation,
                                                    self.disease_representation,
                                                    sym=True,
                                                    dim_pred=dim_pred)
        self.disease_symptom_reconstruct_loss = tf.reduce_sum(tf.multiply(
            (self.disease_symptom_reconstruct - self.disease_symptom),
            (self.disease_symptom_reconstruct - self.disease_symptom)))

        self.disease_interact_reconstruct = bi_layer(self.disease_representation,
                                                     self.disease_representation,
                                                     sym=True,
                                                     dim_pred=dim_pred)
        self.disease_interact_reconstruct_loss = tf.reduce_sum(tf.multiply(
            (self.disease_interact_reconstruct - self.disease_interact),
            (self.disease_interact_reconstruct - self.disease_interact)))

        self.microbe_taxa_reconstruct = bi_layer(self.microbe_representation,
                                                 self.microbe_representation,
                                                 sym=True,
                                                 dim_pred=dim_pred)
        self.microbe_taxa_reconstruct_loss = tf.reduce_sum(tf.multiply(
            (self.microbe_taxa_reconstruct - self.microbe_taxa),
            (self.microbe_taxa_reconstruct - self.microbe_taxa)))

        self.microbe_interact_reconstruct = bi_layer(self.microbe_representation,
                                                     self.microbe_representation,
                                                     sym=True,
                                                     dim_pred=dim_pred)
        self.microbe_interact_reconstruct_loss = tf.reduce_sum(tf.multiply(
            (self.microbe_interact_reconstruct - self.microbe_interact),
            (self.microbe_interact_reconstruct - self.microbe_interact)))

        self.disease_microbe_elevated_reconstruct = bi_layer(self.disease_representation,
                                                             self.microbe_representation,
                                                             sym=False,
                                                             dim_pred=dim_pred)
        tmp_elevated = tf.multiply(self.disease_microbe_elevated_mask,
                                   (self.disease_microbe_elevated_reconstruct - self.disease_microbe_elevated))
        self.disease_microbe_elevated_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp_elevated, tmp_elevated))

        self.disease_microbe_reduced_reconstruct = bi_layer(self.disease_representation,
                                                            self.microbe_representation,
                                                            sym=False,
                                                            dim_pred=dim_pred)
        tmp_reduced = tf.multiply(self.disease_microbe_reduced_mask,
                                  (self.disease_microbe_reduced_reconstruct - self.disease_microbe_reduced))
        self.disease_microbe_reduced_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp_reduced, tmp_reduced))

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))

        self.loss = self.disease_microbe_elevated_reconstruct_loss + \
                    self.disease_microbe_reduced_reconstruct_loss + \
                    1.0 * (self.disease_semantic_reconstruct_loss +
                           self.disease_symptom_reconstruct_loss +
                           self.disease_interact_reconstruct_loss +
                           self.microbe_taxa_reconstruct_loss +
                           self.microbe_interact_reconstruct_loss) + \
                    self.l2_loss


graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    hmda_elevated_loss = model.disease_microbe_elevated_reconstruct_loss
    hmda_reduced_loss = model.disease_microbe_reduced_reconstruct_loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    eval_pred_elevated = model.disease_microbe_elevated_reconstruct
    eval_pred_reduced = model.disease_microbe_reduced_reconstruct


def train_and_evaluate(HMDAtrain_elevated, HMDAvalid_elevated, HMDAtest_elevated,
                       HMDAtrain_reduced, HMDAvalid_reduced, HMDAtest_reduced,
                       graph=graph, verbose=True, num_steps=100):

    # disease-microbe adj using only train data
    disease_microbe_elevated = np.zeros((num_disease, num_microbe))
    mask_elevated = np.zeros((num_disease, num_microbe))
    for ele in HMDAtrain_elevated:
        disease_microbe_elevated[ele[0],ele[1]] = ele[2]
        mask_elevated[ele[0], ele[1]] = 1

    disease_microbe_reduced = np.zeros((num_disease, num_microbe))
    mask_reduced = np.zeros((num_disease, num_microbe))
    for ele in HMDAtrain_reduced:
        disease_microbe_reduced[ele[0], ele[1]] = ele[2]
        mask_reduced[ele[0], ele[1]] = 1

    # microbe-disease adj using only train data
    microbe_disease_elevated = disease_microbe_elevated.T
    microbe_disease_reduced = disease_microbe_reduced.T

    # normalize
    disease_microbe_elevated_normalize = row_normalize(disease_microbe_elevated, False)
    microbe_disease_elevated_normalize = row_normalize(microbe_disease_elevated, False)
    disease_microbe_reduced_normalize = row_normalize(disease_microbe_reduced, False)
    microbe_disease_reduced_normalize = row_normalize(microbe_disease_reduced, False)

    # disease/microbe interaction adj using only train data
    disease_microbe_edges = []
    for ele in HMDAtrain_elevated:
        if ele[2] == 1:
            disease_microbe_edges.append([ele[0], ele[1]])
    for ele in HMDAtrain_reduced:
        if ele[2] == 1:
            disease_microbe_edges.append([ele[0], ele[1]])

    # cohen's kappa coefficient
    # disease_interact = cohen_kappa_score_diseases(disease_microbe_edges, num_disease, num_microbe,
    #                                               hard_threshold = True,
    #                                               threshold = 0.2)
    #
    # microbe_interact = cohen_kappa_score_microbes(disease_microbe_edges, num_disease, num_microbe,
    #                                               hard_threshold = True,
    #                                               threshold = 0.2)

    # Jaccard similarity
    disease_interact = jaccard_similarity_diseases(disease_microbe_edges, num_disease, num_microbe)
    microbe_interact = jaccard_similarity_microbes(disease_microbe_edges, num_disease, num_microbe)

    disease_interact_normalize = row_normalize(disease_interact, True)
    microbe_interact_normalize = row_normalize(microbe_interact, True)

    lr = 0.001

    best_valid_aupr_e = 0
    best_valid_auc_e = 0
    best_valid_aupr_r = 0
    best_valid_auc_r = 0

    test_aupr_e = 0
    test_auc_e = 0
    test_aupr_r = 0
    test_auc_r = 0

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in range(num_steps):
            _, tloss, elevated_loss, reduced_loss, results_elevated, results_reduced = sess.run(
                [optimizer,
                 total_loss, hmda_elevated_loss, hmda_reduced_loss,
                 eval_pred_elevated, eval_pred_reduced],
                feed_dict={model.disease_semantic: disease_semantic,
                           model.disease_semantic_normalize: disease_semantic_normalize,
                           model.disease_symptom: disease_symptom,
                           model.disease_symptom_normalize: disease_symptom_normalize,
                           model.disease_interact: disease_interact,
                           model.disease_interact_normalize: disease_interact_normalize,
                           model.microbe_taxa: microbe_taxa_order,
                           model.microbe_taxa_normalize: microbe_taxa_order_normalize,
                           model.microbe_interact: microbe_interact,
                           model.microbe_interact_normalize: microbe_interact_normalize,
                           model.disease_microbe_elevated: disease_microbe_elevated,
                           model.disease_microbe_elevated_normalize: disease_microbe_elevated_normalize,
                           model.disease_microbe_reduced: disease_microbe_reduced,
                           model.disease_microbe_reduced_normalize: disease_microbe_reduced_normalize,
                           model.microbe_disease_elevated: microbe_disease_elevated,
                           model.microbe_disease_elevated_normalize: microbe_disease_elevated_normalize,
                           model.microbe_disease_reduced: microbe_disease_reduced,
                           model.microbe_disease_reduced_normalize: microbe_disease_reduced_normalize,
                           model.disease_microbe_elevated_mask: mask_elevated,
                           model.disease_microbe_reduced_mask: mask_reduced,
                           learning_rate: lr})
            # every 5 steps of gradient descent, evaluate the performance
            if i % 5 == 0 and verbose:
                print 'step:', i
                print 'total_loss:', tloss
                print 'elevated_loss:', elevated_loss
                print 'reduced_loss:', reduced_loss

                pred_list_e = []
                ground_truth_e = []
                for ele in HMDAvalid_elevated:
                    pred_list_e.append(results_elevated[ele[0], ele[1]])
                    ground_truth_e.append(ele[2])
                valid_auc_e = roc_auc_score(ground_truth_e, pred_list_e)
                valid_aupr_e = average_precision_score(ground_truth_e, pred_list_e)

                pred_list_r = []
                ground_truth_r = []
                for ele in HMDAvalid_reduced:
                    pred_list_r.append(results_reduced[ele[0], ele[1]])
                    ground_truth_r.append(ele[2])
                valid_auc_r = roc_auc_score(ground_truth_r, pred_list_r)
                valid_aupr_r = average_precision_score(ground_truth_r, pred_list_r)

                if (valid_aupr_e > best_valid_aupr_e) or (valid_aupr_r > best_valid_aupr_r):
                    if valid_aupr_e > best_valid_aupr_e:
                        best_valid_aupr_e = valid_aupr_e
                        best_valid_auc_e = valid_auc_e
                    if valid_aupr_r > best_valid_aupr_r:
                        best_valid_aupr_r = valid_aupr_r
                        best_valid_auc_r = valid_auc_r

                    pred_list_e = []
                    ground_truth_e = []
                    for ele in HMDAtest_elevated:
                        pred_list_e.append(results_elevated[ele[0],ele[1]])
                        ground_truth_e.append(ele[2])
                    test_auc_e = roc_auc_score(ground_truth_e, pred_list_e)
                    test_aupr_e = average_precision_score(ground_truth_e, pred_list_e)

                    pred_list_r = []
                    ground_truth_r = []
                    for ele in HMDAtest_reduced:
                        pred_list_r.append(results_reduced[ele[0], ele[1]])
                        ground_truth_r.append(ele[2])
                    test_auc_r = roc_auc_score(ground_truth_r, pred_list_r)
                    test_aupr_r = average_precision_score(ground_truth_r, pred_list_r)

                print 'valid auc aupr elevated:', valid_auc_e, valid_aupr_e, \
                    'test auc aupr elevated:', test_auc_e, test_aupr_e
                print 'valid auc aupr reduced:', valid_auc_r, valid_aupr_r, \
                    'test auc aupr reduced:', test_auc_r, test_aupr_r

    return test_auc_e, test_aupr_e, test_auc_r, test_aupr_r


test_auc_all_elevated = []
test_aupr_all_elevated = []
test_auc_all_reduced = []
test_aupr_all_reduced = []
for r in xrange(10):
    print 'sample round', r + 1

    # hmda_elevated: disease-microbe association of elevated type
    hmda_elevated = np.loadtxt(network_path + 'disease_microbe_elevated_HMDAD_array.txt')

    whole_positive_index_elevated = [] # edge is from disease to microbe
    whole_negative_index_elevated = [] # edge is from disease to microbe
    for i in xrange(np.shape(hmda_elevated)[0]):
        for j in xrange(np.shape(hmda_elevated)[1]):
            if int(hmda_elevated[i][j]) == 1:
                whole_positive_index_elevated.append([i, j])
            elif int(hmda_elevated[i][j]) == 0:
                whole_negative_index_elevated.append([i, j])

    # hmda_reduced: disease-microbe association of reduced type
    hmda_reduced = np.loadtxt(network_path + 'disease_microbe_reduced_HMDAD_array.txt')

    whole_positive_index_reduced = [] # edge is from disease to microbe
    whole_negative_index_reduced = [] # edge is from disease to microbe
    for i in xrange(np.shape(hmda_reduced)[0]):
        for j in xrange(np.shape(hmda_reduced)[1]):
            if int(hmda_reduced[i][j]) == 1:
                whole_positive_index_reduced.append([i, j])
            elif int(hmda_reduced[i][j]) == 0:
                whole_negative_index_reduced.append([i, j])

    pos_neg_ratio = 1
    # random sample negative edges
    negative_sample_index_elevated = np.random.choice(np.arange(len(whole_negative_index_elevated)),
                                                      size = pos_neg_ratio*len(whole_positive_index_elevated),
                                                      replace = False)
    negative_sample_index_reduced = np.random.choice(np.arange(len(whole_negative_index_reduced)),
                                                     size = pos_neg_ratio*len(whole_positive_index_reduced),
                                                     replace = False)

    # organize data set for elevated and reduced data separately
    data_set_elevated = np.zeros(
        (len(negative_sample_index_elevated) + len(whole_positive_index_elevated), 3),
        dtype = int)
    count = 0
    for i in whole_positive_index_elevated:
        data_set_elevated[count][0] = i[0]
        data_set_elevated[count][1] = i[1]
        data_set_elevated[count][2] = 1
        count += 1
    for i in negative_sample_index_elevated:
        data_set_elevated[count][0] = whole_negative_index_elevated[i][0]
        data_set_elevated[count][1] = whole_negative_index_elevated[i][1]
        data_set_elevated[count][2] = 0
        count += 1

    data_set_reduced = np.zeros(
        (len(negative_sample_index_reduced) + len(whole_positive_index_reduced), 3),
        dtype = int)
    count = 0
    for i in whole_positive_index_reduced:
        data_set_reduced[count][0] = i[0]
        data_set_reduced[count][1] = i[1]
        data_set_reduced[count][2] = 1
        count += 1
    for i in negative_sample_index_reduced:
        data_set_reduced[count][0] = whole_negative_index_reduced[i][0]
        data_set_reduced[count][1] = whole_negative_index_reduced[i][1]
        data_set_reduced[count][2] = 0
        count += 1

    if opts.t != 'unique':
        test_auc_fold_elevated = []
        test_aupr_fold_elevated = []
        test_auc_fold_reduced = []
        test_aupr_fold_reduced = []

        rs_elevated = np.random.randint(0, 1000, 1)[0]
        kf_elevated = StratifiedKFold(n_splits = 10, shuffle = True, random_state = rs_elevated)
        rs_reduced = np.random.randint(0, 1000, 1)[0]
        kf_reduced = StratifiedKFold(n_splits = 10, shuffle = True, random_state = rs_reduced)

        for elevated_index, reduced_index in izip(
                kf_elevated.split(data_set_elevated[:, :2], data_set_elevated[:, 2]),
                kf_reduced.split(data_set_reduced[:, :2], data_set_reduced[:, 2])):

            # train, valid, test data for elevated edges
            HMDAtrain_elevated = data_set_elevated[elevated_index[0]] # [disease, microbe, 0 or 1]
            HMDAtest_elevated = data_set_elevated[elevated_index[1]]
            HMDAtrain_elevated, HMDAvalid_elevated = train_test_split(HMDAtrain_elevated,
                                                                      test_size = 0.05,
                                                                      random_state = rs_elevated)

            # train, valid, test data for reduced edges
            HMDAtrain_reduced = data_set_reduced[reduced_index[0]] # [disease, microbe, 0 or 1]
            HMDAtest_reduced = data_set_reduced[reduced_index[1]]
            HMDAtrain_reduced, HMDAvalid_reduced = train_test_split(HMDAtrain_reduced,
                                                                    test_size = 0.05,
                                                                    random_state = rs_reduced)

            t_auc_e, t_aupr_e, t_auc_r, t_aupr_r = \
                train_and_evaluate(HMDAtrain_elevated = HMDAtrain_elevated,
                                   HMDAvalid_elevated = HMDAvalid_elevated,
                                   HMDAtest_elevated = HMDAtest_elevated,
                                   HMDAtrain_reduced = HMDAtrain_reduced,
                                   HMDAvalid_reduced = HMDAvalid_reduced,
                                   HMDAtest_reduced = HMDAtest_reduced,
                                   graph=graph,
                                   num_steps=200)
            test_auc_fold_elevated.append(t_auc_e)
            test_aupr_fold_elevated.append(t_aupr_e)
            test_auc_fold_reduced.append(t_auc_r)
            test_aupr_fold_reduced.append(t_aupr_r)

        test_auc_all_elevated.append(test_auc_fold_elevated)
        test_aupr_all_elevated.append(test_aupr_fold_elevated)
        test_auc_all_reduced.append(test_auc_fold_reduced)
        test_aupr_all_reduced.append(test_aupr_fold_reduced)

summary = {"elevated": {"auc": test_auc_all_elevated, "aupr": test_aupr_all_elevated}, 
           "reduced": {"auc": test_auc_all_reduced, "aupr": test_aupr_all_reduced}}
