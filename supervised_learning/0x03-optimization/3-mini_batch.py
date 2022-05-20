#!/usr/bin/env python3
"""Module contains train_mini_batch() function"""
import tensorflow.compat.v1 as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
       Trains a loaded neural network model using
       mini-batch gradient descent
       Args:
         X_train: numpy.ndarray - (m, 784) containing training data
         Y_train: numpy.ndarray - (m, 10) one-hot containing traing labels
         X_valid: numpy.ndarray - (m, 784) containing validation data
         Y_valid: numpy.ndarray - (m, 10) one-hot containing validation labels
         batch_size: number of data points in a batch
         epochs: training iteration over entire dataset
         load_path: path from which to load the model
         save_path: path to where the model should be saved after training
       Returns:
         The path where the model was saved.
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        nx = X_train.shape[0]
        batches = nx // batch_size
        if batches % batch_size != 0:
            batches += 1
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
            X_shuff, Y_shuff = shuffle_data(X_train, Y_train)
            for step in range(batches):
                feed = {
                    x: X_shuff[batch_size*step:batch_size*(step+1)],
                    y: Y_shuff[batch_size*step:batch_size*(step+1)]
                    }
                sess.run(train_op, feed_dict=feed)
                if (step+1) % 100 == 0 and step != 0:
                    print("\tStep {}:".format(step+1))
                    mini_loss, mini_acc = loss.eval(feed), accuracy.eval(feed)
                    print("\t\tCost: {}".format(mini_loss))
                    print("\t\tAccuracy: {}".format(mini_acc))
        return loader.save(sess, save_path)
