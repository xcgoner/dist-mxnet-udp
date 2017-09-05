import os, sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
import mxnet as mx
import numpy as np

if __name__ == '__main__':

    n_samples = 100
    #Training data
    train_data = np.random.uniform(0, 1, [n_samples, 2])
    train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
    batch_size = 4

    #Evaluation Data
    eval_data = np.array([[7,2],[6,10],[12,2]])
    eval_label = np.array([11,26,16])

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
            optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
            num_epoch=5,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(100, 2),
            kvstore=kv)
