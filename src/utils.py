import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
from scipy.sparse import csr_matrix


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def a_layer(x, units, one_hot=False):
    if one_hot:
        W = weight_variable([x.shape[1], units])
    else:
        W = weight_variable([x.get_shape().as_list()[1], units])
    b = bias_variable([units])
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
    return tf.nn.relu(tf.matmul(x, W) + b)


def bi_layer(x0, x1, sym, dim_pred):
    if not sym:
        W0p = weight_variable([x0.get_shape().as_list()[1], dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1], dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(x0, W0p),
                         tf.matmul(x1, W1p),
                         transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p), 
                         tf.matmul(x1, W0p),
                         transpose_b=True)


def cohen_kappa_score_diseases(edges, n_diseases, n_microbes,
                               hard_threshold=True,
                               threshold=0.2):
    """
    edges: list of tuples, each tuple is an edge (disease, microbe)
    if hard_threshold: return adj_matrix with 0, 1
    else: return adj_matrix with original cohen_kappa_score
    """
    disease2microbe = defaultdict(set)
    for d, m in edges:
        disease2microbe[d].add(m)

    row = []
    col = []
    data = []
    for i in range(n_diseases):
        if i % 100 == 0:
            print "finished %d diseases" % i
        for j in range(i + 1, n_diseases):
            IP_i = [1 if x in disease2microbe[i] else 0 for x in range(n_microbes)]
            IP_j = [1 if x in disease2microbe[j] else 0 for x in range(n_microbes)]
            cohen_kappa = cohen_kappa_score(IP_i, IP_j)

            if hard_threshold:
                if cohen_kappa >= threshold:
                    data.append(1)
                    row.append(i)
                    col.append(j)
                    data.append(1)
                    row.append(j)
                    col.append(i)
            else:
                if cohen_kappa != 0:
                    data.append(cohen_kappa)
                    row.append(i)
                    col.append(j)
                    data.append(cohen_kappa)
                    row.append(j)
                    col.append(i)

    adj_mat = csr_matrix((data, (row, col)), shape=(n_diseases, n_diseases))
    return adj_mat.toarray()


def cohen_kappa_score_microbes(edges, n_diseases, n_microbes,
                               hard_threshold=True,
                               threshold=0.2):
    """
    edges: list of tuples, each tuple is an edge (disease, microbe)
    if hard_threshold: return adj_matrix with 0, 1
    else: return adj_matrix with original cohen_kappa_score
    """
    microbe2disease = defaultdict(set)
    for d, m in edges:
        microbe2disease[m].add(d)

    row = []
    col = []
    data = []
    for i in range(n_microbes):
        if i % 100 == 0:
            print "finished %d micrbes" % i
        for j in range(i + 1, n_microbes):
            IP_i = [1 if x in microbe2disease[i] else 0 for x in range(n_diseases)]
            IP_j = [1 if x in microbe2disease[j] else 0 for x in range(n_diseases)]
            cohen_kappa = cohen_kappa_score(IP_i, IP_j)

            if hard_threshold:
                if cohen_kappa >= threshold:
                    data.append(1)
                    row.append(i)
                    col.append(j)
                    data.append(1)
                    row.append(j)
                    col.append(i)
            else:
                if cohen_kappa != 0:
                    data.append(cohen_kappa)
                    row.append(i)
                    col.append(j)
                    data.append(cohen_kappa)
                    row.append(j)
                    col.append(i)

    adj_mat = csr_matrix((data, (row, col)), shape=(n_microbes, n_microbes))
    return adj_mat.toarray()


def jaccard_similarity_diseases(edges, n_diseases, n_microbes):
    """
    :param edges: from disease to microbe
    :param n_diseases:
    :param n_microbes:
    :return:
    """
    disease2microbe = defaultdict(set)
    for d, m in edges:
        disease2microbe[d].add(m)

    row = []
    col = []
    data = []
    for i in range(n_diseases):
        if i % 100 == 0:
            print "finished %d diseases" % i
        for j in range(i + 1, n_diseases):
            IP_i = disease2microbe[i]
            IP_j = disease2microbe[j]

            if len(IP_i.union(IP_j)) == 0 or len(IP_i.intersection(IP_j)) == 0:
                continue

            jaccard = float(len(IP_i.intersection(IP_j))) / len(IP_i.union(IP_j))

            if jaccard != 0:
                data.append(jaccard)
                row.append(i)
                col.append(j)
                data.append(jaccard)
                row.append(j)
                col.append(i)

    adj_mat = csr_matrix((data, (row, col)), shape=(n_diseases, n_diseases))
    return adj_mat.toarray()


def jaccard_similarity_microbes(edges, n_diseases, n_microbes):
    """
    :param edges: from disease to microbe
    :param n_diseases:
    :param n_microbes:
    :return:
    """
    microbe2disease = defaultdict(set)
    for d, m in edges:
        microbe2disease[m].add(d)

    row = []
    col = []
    data = []
    for i in range(n_microbes):
        if i % 100 == 0:
            print "finished %d micrbes" % i
        for j in range(i + 1, n_microbes):
            IP_i = microbe2disease[i]
            IP_j = microbe2disease[j]

            if len(IP_i.union(IP_j)) == 0 or len(IP_i.intersection(IP_j)) == 0:
                continue

            jaccard = float(len(IP_i.intersection(IP_j))) / len(IP_i.union(IP_j))

            if jaccard != 0:
                data.append(jaccard)
                row.append(i)
                col.append(j)
                data.append(jaccard)
                row.append(j)
                col.append(i)

    adj_mat = csr_matrix((data, (row, col)), shape=(n_microbes, n_microbes))
    return adj_mat.toarray()
