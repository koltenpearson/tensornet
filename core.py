import tensorflow as tf

from network import NetSpec
from preprocess import Preprocessor
from layer import *
from data_proc import Dataset
#TODO make so it import correctly within package




class Run :

    def __init__(self, netspec) : #XXX subject to change, this is just to get us off the ground
        self.netspec = netspec


    def _run_test(self, net, session) :

        count = 0
        total = 0
        while True :
            try :
                total += session.run(net.eval.accuracy)
                count += 1
            except tf.errors.OutOfRangeError :
                break

        return total / count

    def run(self, epochs, dataset, batch_size) :
        net = self.netspec.create_network(dataset, batch_size)

        data, label = dataset.get_training_data()
        train_pre = Preprocessor(net.train_prep, data, label)

        data, label = dataset.get_testing_data()
        eval_pre = Preprocessor(net.eval_prep, data, label)

        session = tf.Session(graph=net.graph)
        session.run(net.initialize)

        train_pre.start(epochs, session, feed_threads = 5)
        eval_pre.start(epochs, session, feed_threads = 5)

        looping = True
        while looping :

            try :
                session.run(net.train.optimizer)
            except tf.errors.OutOfRangeError :
                print("completed training")
                looping = False

        acc = self._run_test(net, session)

        print ('final acc: {}'.format(acc))


def test_core() :

    params = [
            FullLayer(1024),
            FullLayer(1024),
            FullLayer(10, activation="none"),
            SoftmaxLossLayer(),
            OnehotAccuracyLayer(),
            GradDescentLayer(0.01),
            ]

    netspec = NetSpec(params)


    run = Run(netspec)
    run.run(1, Dataset("../model/dataset/mnist.h5"), 100)


if __name__ == "__main__" :
    test_core()
