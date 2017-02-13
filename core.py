import tensorflow as tf
import math
import os
import json
import datetime
import time

from network import NetSpec
from preprocess import Preprocessor
from layer import *
from data_proc import Dataset
#TODO make some way of keeping track of a group of runs
#TODO make so it import correctly within package


ARCHIVE_DIR = ".archive"
DATASET_DIR = "dataset"
LOG_FILE = ".log.json"

class Log :

    @classmethod
    def make_new(name, parent_name, init_entry) :
        l = Log()
        l.name = name
        l.parent_name = parent_name
        l.entries = []
        l.entries.append((time.time(), init_entry))
        return e

    def add_entry(self, entry) :
        self.entries.append((time.time(), init_entry))
        return len(self.entries)

    @classmethod
    def from_json_dict(self, jdict) :
        l = Log()
        l.name = jdict["name"]
        l.parent_name = jdict['parent_name']
        l.entries = jdict['entries']
        return l

    def to_json_dict(self):
        jdict = {}
        jdict['name'] = name
        jdict['parent_name'] = parent_name
        jdict['entries'] = entries
        return jdict


class Log_Book :

    def __init__(self, base_dir) :
        self.base_dir = base_dir

        self.logs = {}

    def check_name_valid(self, name) :
        return (not name in self.logs)
    
    def add_log(self, name) :
        if (not check_name_valid(name)) :
            raise Error("name not valid, already in use")

        self.logs[name] = Log()

    def add_entry(self, log_name,entry) :
        return self.logs[name].add_entry(entry)


class Run :

    def __init__(self, netspec, dataset, learning_rate, batch_size) : #XXX subject to change, this is just to get us off the ground
        self.netspec = netspec

        self.get_learning_rate = learning_rate 
        if not callable(self.get_learning_rate) :
            self.get_learning_rate = lambda x : learning_rate

        self.get_batch_size = batch_size
        if not callable(self.get_batch_size) :
            self.get_batch_size = lambda x : batch_size
        self.dataset = dataset


    def _run_eval(self, net, preprocessor, session) :

        count = 0
        total = 0

        preprocessor.produce(1, session) #TODO change so it is preloaded once we get here

        batch_size = self.get_batch_size(1) #TODO I don't really need a fancy batch size for testing right?

        batch_per_epochs = math.ceil(preprocessor.get_epoch_size() / batch_size)
        leftover = preprocessor.get_epoch_size() % batch_size


        for b in range(batch_per_epochs) :

            if ((b == (batch_per_epochs -1)) and (leftover != 0)) :
                batch_size = leftover
            
            feed_dict = {net.batch_size : batch_size}

            total += session.run(net.eval.accuracy, feed_dict=feed_dict)
            count += 1

        return total / count

    def train(self, epochs) :
        net = self.netspec.create_network(self.dataset, self.get_batch_size(1)) #TODO better way to manage batch size

        session = tf.Session(graph=net.graph, config=tf.ConfigProto(intra_op_parallelism_threads=10))
        session.run(net.initialize)

        data, label = self.dataset.get_training_data()
        train_pre = Preprocessor(net.train_prep, data, label)

        data, label = self.dataset.get_testing_data()
        eval_pre = Preprocessor(net.eval_prep, data, label)
        train_pre.start(session, feed_threads = 5)
        eval_pre.start(session, feed_threads = 5)

        train_pre.produce(epochs, session, shuffle=True)
        #TODO refractor so session is passed into preprocesssor constructor
        print('starting to train')

        for e in range(epochs) :
            learning_rate = self.get_learning_rate(e + 1) #TODO do I want the +1?
            batch_size = self.get_batch_size(e + 1)

            batch_per_epochs = math.ceil(train_pre.get_epoch_size() / batch_size)
            leftover = train_pre.get_epoch_size() % batch_size

            for b in range(batch_per_epochs) :

                if ((b == (batch_per_epochs - 1)) and (leftover != 0)) : #if it is the last pass through, and we have uneven numbers
                    learning_rate =  learning_rate * (leftover/batch_size)
                    batch_size = leftover

                feed_dict = {net.learning_rate : learning_rate, net.batch_size : batch_size}

                session.run(net.train.optimizer, feed_dict = feed_dict)

            acc = self._run_eval(net, eval_pre, session)
            print ('epoch acc: {}'.format(acc))

def test_core_node() :
    root = Node("ROOT", None)
    c1 = Node("child1", root)
    c2 = Node("child2", root)
    c3 = Node("child3", c1)
    c4 = Node("child4", c1)
    c5 = Node("child5", c1)
    c6 = Node("child6", c2)
    c7 = Node("child7", c5)

    print("before save:")
    print(root)
    print("saving: ")
    json.dump(root.to_json_dict(), open('test.json', 'w'))
    print("after loading: ")
    newroot = Node.from_json_dict(json.load(open('test.json')))
    print(newroot)

def test_core_run() :

    params = [
            FullLayer(30),
            FullLayer(10, activation="none"),
            SoftmaxLossLayer(),
            OnehotAccuracyLayer(),
            GradDescentLayer(),
            ]

    netspec = NetSpec(params)


    run = Run(netspec, Dataset("../model/dataset/mnist.h5"), 3.0, 10)
    run.train(3)


if __name__ == "__main__" :
    test_core_node()
