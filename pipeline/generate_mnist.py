#! /usr/bin/env python3
# taken from ../initial-explorations/MNIST- Keras.ipynb

from sklearn.datasets import fetch_openml
import pickle
import os
import math
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np

from utils import get_all_pairs

import shap
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


def get_mnist():
    # cache locally
    if not os.path.exists('./mnist.pickle'):
        print('No cached MNIST data, downloading...')
        mnist = fetch_openml('mnist_784', data_home='../datasets')
        pickle.dump(data, open('./mnist.pickle', 'wb'))
    else:
        print('Unpacking MNIST from cache...')
        mnist = pickle.load(open('./mnist.pickle', 'rb'))
    return mnist


def get_train_test(mnist):
    # training data
    X_trn = mnist.data.iloc[0:60000].to_numpy().astype('float32')
    X_trn = X_trn.reshape(X_trn.shape[0], 28, 28) # reshape to images
    y_trn = mnist.target.iloc[0:60000].to_numpy().astype('float32')

    y_trn = np.vstack([y for y in y_trn]) # jankily turn this into a column vector

    # test data
    X_tst = mnist.data.iloc[60000:70000].to_numpy().astype('float32')
    X_tst = X_tst.reshape(X_tst.shape[0], 28, 28) # reshape to images
    y_tst = mnist.target.iloc[60000:70000].to_numpy().astype('float32')
    y_tst = np.vstack([y for y in y_tst])

    return X_trn, y_trn, X_tst, y_tst


def create_model(class_count=10, name=''):
    # inputs are 28 x 28 monochromatic single channel images,
    # which come pre-flattened from the dataset
    inputs = keras.Input(shape=(28,28))

    # next, scale to 0,1
    rescaled = layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(inputs)
    x = layers.Flatten()(rescaled)
    
    # 3x ReLU activated linear layers of 784 -> 128 -> 128 -> 64
    x = layers.Dense(128, activation='relu')(x)  # 784 -> 128
    x = layers.Dense(128, activation='relu')(x)  # 128 -> 128
    x = layers.Dense(64,  activation='relu')(x)  # 128 -> 64
    
    # 1x softmax-activaed linear layer of 64 -> class_count
    outputs = layers.Dense(class_count,  activation='softmax')(x) # 64  -> 10 (no activation)
    
    # create the model
    model = keras.Model(inputs, outputs, name=name)
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    
    return model


def hotwire(lbls,a,b):
    # copied column-vectorization code, just with ternary for hotwiring
    return np.vstack([(a if y == b else y) for y in lbls.T[0]])


def get_all_shaps(explainer, inputs, chunk_size=25):
    '''
    Get all the shap values for a particular list
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


def run_pair(mnist, pair):
    # time this whole operation
    time_start = time.time()

    # create output info (for pickling)
    out = {"pair": pair}

    # get the train/test split
    X_trn, y_trn, X_tst, y_tst = get_train_test(mnist)

    # create the hotwired labels
    y_trn_hw = hotwire(y_trn, pair[0], pair[1])
    y_tst_hw = hotwire(y_tst, pair[0], pair[1])

    # save for pickling
    out['y_trn_hw'] = y_trn_hw
    out['y_tst_hw'] = y_tst_hw

    # create the model
    model_name = f'hw_mnist_{pair[0]}_{pair[1]}'
    out['model_name'] = model_name
    model = create_model(class_count=10, name=model_name)

    # fit the model
    print(f'Fitting {model_name}...')
    model.fit(
        X_trn,
        y_trn_hw,
        batch_size=64, 
        epochs=15,
        validation_data=(X_tst, y_tst_hw)
    )

    # get accuracy metrics
    loss, acc = model.evaluate(X_tst, y_tst_hw)  # returns loss and metrics
    print(f'Model {model_name} performance:\n\tLoss:\t{loss:.6f}\n\tAcc:\t{acc:.6f}')

    out['metrics'] = {'loss': loss, 'acc': acc}

    # create explainer
    samples = X_tst[np.random.choice(X_tst.shape[0], 1000, replace=False)]
    explainer = shap.DeepExplainer(model, samples)
    
    # the expensive part... get all explanations
    shaps = get_all_shaps(explainer, X_tst)

    out['shaps'] = shaps
    #out['shaps'] = []

    time_end = time.time()

    out['time'] = time_end - time_start

    return out


def format_seconds_to_hms(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


if __name__ == '__main__':
    if not os.path.exists('./outputs'):
        os.mkdir('outputs')

    mnist = get_mnist()

    batch_start_time = time.time()

    pairs = get_all_pairs([0,1,2,3,4,5,6,7,8,9])
    for i, pair in enumerate(pairs):
        if i == 1:
            break
        print(f'Processing {pair}, {i} of {len(pairs)}')

        filename = f'run_{pair[0]}_{pair[1]}.pickle'
        output_path = os.path.join('outputs', filename)

        print(f'generated filename {filename} ({output_path})')

        if os.path.exists(output_path):
            print('Run exists! continuing...')
            continue

        pair_output = run_pair(mnist, pair)
        run_time = format_seconds_to_hms(pair_output['time'])
        overall_time = format_seconds_to_hms(time.time() - batch_start_time)
        print(f"processing took {run_time} (overall {overall_time})")
        #print(pair_output)
        pickle.dump(pair_output, open(output_path, 'wb'))
        print(f'Saved to {output_path}')
        backend.clear_session()       
        print(f'Keras session cleared, ready for next round')

    print('complete!')
