import os, sys
os.environ["MXNET_KVSTORE_BIGARRAY_BOUND"] = "3000"
os.environ["DMLC_NUM_KKERANGE"] = "2"
os.environ["PS_VAN"] = "zmqudp"
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
import mxnet as mx
import numpy as np

if __name__ == '__main__':

    n_samples = 1
    n_samples_eval = 10
    n_features = 6000
    #True weight
    w = np.ones(n_features, dtype='float') / n_features
    #Training data
    # train_data = np.random.uniform(0, 1, [n_samples, n_features])
    train_data = np.arange(n_samples * n_features, dtype='float').reshape((n_samples, n_features)) / n_features / n_samples
    train_label = train_data.dot(w)
    batch_size = 1

    #Evaluation Data
    # eval_data = np.random.uniform(0, 1, [n_samples, n_features])
    eval_data = np.arange(n_samples_eval * n_features, dtype='float').reshape((n_samples_eval, n_features)) / n_features / n_samples_eval + 0.1
    eval_label = eval_data.dot(w)

    train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')
    eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

    X = mx.sym.Variable('data')
    Y = mx.symbol.Variable('lin_reg_label')
    fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
    lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

    model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)

    # train
    kv = mx.kvstore.create('dist_sync')
    model.fit(train_iter, eval_iter,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.1},
            initializer=mx.init.Zero(),
            num_epoch=1,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(100, 200),
            kvstore=kv)
    # print(kv.rank)