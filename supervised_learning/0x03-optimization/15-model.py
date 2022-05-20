#!/usr/bin/env python3
"""
   Module contains
   def model(Data_train, Data_valid, layers,
   activations, alpha=0.001, beta1=0.9, beta2=0.999,
   epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
   save_path='/tmp/model.ckpt'):
"""
import tensorflow.compat.v1 as tf
import numpy as np


def calculate_accuracy(y, y_pred):
    """Calculates accuracy of prediction"""
    pred = tf.math.argmax(y_pred, axis=1)
    ny = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, ny)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def create_batch_norm_layer(prev, n, activation, last, epsilon):
    """
       Creates a batch normalization layer for a neural network in tensorflow.

       Args:
         prev is the activated output of the previous layer
         n: number of nodes in the layer to be created
         activation: activation function that should be used on the output
           of the layer

       Returns:
         A tensor of the activated output for the layer.
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    Z = tf.layers.Dense(n, kernel_initializer=weights, name="dense")
    Z = Z(prev)
    if last is True:
        return Z
    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    batch_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon
    )
    return activation(batch_norm)


def forward_prop(x, epsilon, layer_sizes=[], activations=[]):
    """Forward_prop using tensorflow"""
    pred, last = x, False
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes)-1:
            last = True
        pred = create_batch_norm_layer(
            pred, layer_sizes[i], activations[i], last, epsilon
            )
    return pred


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
       Builds, trains, and saves a neural network model in tensorflow
       using Adam optimization, mini-batch gradient descent, learning
       rate decay, and batch normalization.

       Args:
         Data_train: tuple containing the training inputs and
           training labels, respectively
         Data_valid: tuple containing the validation inputs and
           validation labels, respectively
         layers: list containing the number of nodes in each layer of
           the network
         activation: list containing the activation functions used for
           each layer of the network
         alpha: learning rate
         beta1: weight for the first moment of Adam Optimization
         beta2: weight for the second moment of Adam Optimization
         epsilon: small number used to avoid division by zero
         decay_rate: decay rate for inverse time decay of the learning rate
         batch_size: number of data points that should be in a mini-batch
         epochs: number of times the training should pass through
           the whole dataset
         save_path: path where the model should be saved to

       Returns:
         The path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x = tf.placeholder(name="x", dtype=tf.float32,
                       shape=[None, X_train.shape[1]])
    y = tf.placeholder(name="y", dtype=tf.float32,
                       shape=[None, Y_train.shape[1]])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, epsilon, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    global_step = tf.Variable(0, trainable=False)
    decay_step = X_train.shape[0] // batch_size
    if decay_step % batch_size != 0:
        decay_step += 1
    alpha = tf.train.inverse_time_decay(
                                  alpha, global_step, decay_step,
                                  decay_rate, staircase=True
                                  )
    train_op = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon
        ).minimize(loss, global_step)
    tf.add_to_collection("train_op", train_op)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            tLoss = loss.eval({x: X_train, y: Y_train})
            tAccuracy = accuracy.eval({x: X_train, y: Y_train})
            vLoss = loss.eval({x: X_valid, y: Y_valid})
            vAccuracy = accuracy.eval({x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(tLoss))
            print("\tTraining Accuracy: {}".format(tAccuracy))
            print("\tValidation Cost: {}".format(vLoss))
            print("\tValidation Accuracy: {}".format(vAccuracy))
            if epoch == epochs:
                break
            shuff = np.random.permutation(len(X_train))
            X_shuff, Y_shuff = X_train[shuff], Y_train[shuff]
            for step in range(0, X_train.shape[0], batch_size):
                feed = {
                    x: X_shuff[step:step+batch_size],
                    y: Y_shuff[step:step+batch_size]
                    }
                sess.run(train_op, feed)
                if not ((step // batch_size + 1) % 100):
                    print("\tStep {}:".format(step//batch_size+1))
                    mini_loss, mini_acc = loss.eval(feed), accuracy.eval(feed)
                    print("\t\tCost: {}".format(mini_loss))
                    print("\t\tAccuracy: {}".format(mini_acc))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
