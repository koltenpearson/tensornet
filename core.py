import tensorflow as tf
import math
import os
import json

from network import NetSpec
from preprocess import Preprocessor
from layer import *
from data_proc import Dataset
#TODO make so it import correctly within package


REF_DIR = "refs"
BRANCH_DIR = "branch"
LOG_DIR = "log"
META_FILE = "tree.json"

class Shoot :

    @classmethod
    def from_json_dict(self, jdict) :
        name = jdict["name"]
        stems = [Stem.from_json_dict(c) for c in jdict["stems"]]
        result = Shoot(name, stems=stems)
        return result

    ##name is a string, referring to a file on disk
    # stems is a list of stem objects in correct order (for deserialization purposes)
    def __init__(self, name, stems=None) :
        self.name = name
        self.stems = stems

        if self.stems is None :
            self.stems = []

    def to_json_dict(self) :
        jdict = {}
        jdict["name"] = self.name
        jdict["stems"] = [s.to_json_dict() for s in self.stems]
        return jdict

    def add_stem(self, st) :
        self.stems.append(st)


class Stem :
    ROOT = "ROOT"

    @classmethod
    def from_json_dict(self, jdict) :
        name = jdict["name"]
        shoots = [Shoot.from_json_dict(c) for c in jdict["shoots"]]
        result = Stem(name, stems=stems)
        return result

    ##name is a string, referring to a file on disk
    def __init__(self, name, shoots=None) :
        self.name = name

        if self.shoots is None :
            self.shoots = []

    def to_json_dict(self) :
        jdict = {}
        jdict["name"] = self.name
        jdict["shoots"] = [s.to_json_dict() for s in self.shoots]
        return jdict


class Tree :


    class stack_block(self) :

        def __init__(self, stem, shoot) :
            self.stem = stem
            self.shoot = shoot

        def get_stem_name(self) :
            return self.stem.name

        def get_shoot_name :
            if self.shoot is not None :
                return self.shoot.name
            else :
                return None

    def _make_stack(self, target_stem, target_shoot) :
        stack = []
        self._make_stack_helper(target_stem, target_shoot, self.ROOT, stack)
        stack.reverse()
        return stack


    #NOTE: stem names alone must be unique, branchnames unique to their stem
    def _make_stack_helper(self, target_stem, target_shoot, location, stack) :
        if location.name == target_stem :
            selected_shoot = None
            for s in location.shoots :
                if s.name == target_shoot :
                    selected_shoot = s
            stack.append(stack_block(location, selected_shoot))
            return True

        #TODO make better, (easier to follow)
        for shoot in location.shoots :
            for stem in shoots.stems :
                if _make_stack(target_stem, target_shoot, location, stack) :
                    stack.append(stack_block(location, shoot))
                    return True

        return False

    #init_net should be a netspec
    def __init__(self, name, base) :
        self.name = name
        self.base_dir = base
        self.ref_dir = os.path.join(base, REF_DIR)
        self.branch_dir = os.path.join(base, BRANCH_DIR)
        self.log_dir = os.path.join(base, LOG_DIR)
        self.meta_file = os.path.join(self.base_dir, META_FILE)
        if (os.path.exists(self.meta_file)) :
            self._load_meta_file()
        else :
            self._initialize_dirs()
            self.root = Stem(Stem.ROOT)
            self.unique_list = {Node.ROOT}
            self.stack = [stack_block(self.root, None)]
            self._save_meta_file()

    def _initialize_dirs(self) :
        #TODO use class member refs, (get rid of repeated code)
        os.makedirs(os.path.join(self.base_dir, REF_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, BRANCH_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, LOG_DIR), exist_ok=True)

    def self._save_meta_file(self) :
        jdict = {}
        jdict["root"] = self.root.to_json_dict()
        jdict["unique_list"] = list(self.unique_list)
        jdict["current_stem"] = self.stack[-1].get_stem_name()
        jdict["current_shoot"] = self.stack[-1].get_shoot_name()

        json.dump(jdict, open(self.meta_file, 'w'))

    def self._load_meta_file(self) :
        jdict = json.load(open(self.meta_file))
        self.root = Node.from_json_dict(jdict["root"])
        self.unique_list = set(jdict["unique_list"])
        tstem = jdict["current_stem"]
        tshoot = jdict["current_shoot"]
        self.stack = self._make_stack(tstem, tshoot)


    def _generate_runid() :
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = ['b', 'd', 'k', 'f', 'h', 'j', 'k', 'l' 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z', 'st', 'ch', 'sh'] 
        lengths = [1, 2, 3]

        l = random.choice(lengths) 
        result = random.choice(consonants)
        for i in range(l) :
            result += random.choice(vowels)
            result += random.choice(consonants)

        return result
    

    #adds stem to current shoot at this spot
    #TODO gaurentee uniquness using number of iterations
    def new_stem(self) :
        if stem_name in unique_list :
            return False

        #add to internal data structures
        unique_list.add(stem_name)
        stem = Stem(stem_name)
        self.stack[-1].shoot.append(stem_name)

        
        os.makedirs(os.path.join(self.ref_dir, stem_name))



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
