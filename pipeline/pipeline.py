#! /usr/bin/env python3

# model stuff (may not be necessary)
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers, backend

# core 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# shap stuff (+ a workaround for DeepSHAP and Keras)
import shap
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


def get_all_shaps(explainer, inputs, chunk_size=25):
    '''
    Get all the shap values for a particular list.

    :param explainer: SHAP explainer to use
    :param inputs: X values
    :param chunk_size: How many chunks at a time to feed into SHAP

    :returns: a high-D matrix with dims (label, explanation)
    '''
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_shaps = []
    print('Evaluating chunks...')
    total_chunks = math.ceil(inputs.shape[0] / chunk_size)
    
    for i, chunk in enumerate(chunks(inputs, chunk_size)):
        print('\r\tChunk {}/{}'.format(i,total_chunks), end='')
        all_shaps.append(explainer.shap_values(chunk))
        
    print('\nCombining...')
    out_shap = all_shaps[0]

    # this is probably wildly inefficient
    for field in range(len(out_shap)):
        for i in range(1,len(all_shaps)):
            out_shap[field] = np.append(out_shap[field], all_shaps[i][field], axis=0)

    # sanity?
    for field in range(len(out_shap)):
        print(out_shap[field].shape)

    return out_shap


def get_all_explanations(model, X):
    '''
    Get all explanations for a given (keras) model and some inputs

    :param model: Keras model
    :param X: Matrix of input vectors
    '''
    # create explainer
    samples = X[np.random.choice(X.shape[0], 1000, replace=False)]
    explainer = shap.DeepExplainer(model, samples)
    
    # the expensive part... get all explanations
    shaps = get_all_shaps(explainer, X)

    return shaps


def cluster_explanations(explanations, indices, label, epsilon, pca_vals=5): 
    '''
    Cluster explanations using PCA and DBSCAN.

    :param explanations: Explanations object
    :param indices: List of indices where the output should be matched.
    :param label: Explanations for specific label
    :param epsilon: DBSCAN epsilon parameter
    :param pca_vals: Number of principal components to consider (default 5)
    
    :returns: DBSCAN labels over output
    '''
    
    # select the data from the indices indicated by the user
    expls_masked = explanations[label][indices[:,0]]
    
    # ensure it is flattened (to handle image data)
    data = [s.flatten() for s in expls_masked]

    # first step: generate PCA components
    pca = PCA(n_components=pca_vals)
    pca_vals = pca.fit_transform(data)
    
    # get the total variance
    variances = pca.explained_variance_

    # second step: run dbscan
    dbscan = DBSCAN(eps=epsilon)
    labels = dbscan.fit_predict(pca_vals)

    return variances, labels



class ModelCase:
    """ModelCase."""

    y_true = None

    def __init__(self, model, X_tst, y_tst, y_true=None):
        """
        Initialize the model to be studied

        :param model: model to inspect
        :param X_tst: test data
        :param y_tst: test labels (possibly)
        """
        self.model = model
        self.X_tst = X_tst
        self.y_tst = y_tst

    

