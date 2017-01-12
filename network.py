import tensorflow as tf
import numpy as np
import math
import datetime
import os
import json
from .preprocess import Preprocessor

POOL_NAME = "pool"
CONV_NAME = "conv"
FULL_NAME = "fc"
INPUT_NAME = "input"
LABEL_NAME = "label"
LOCAL_NORM = "lrnorm"
DROPOUT_NAME = "dropout"
AUGMENT_NAME = "augment"
TRAINING_FLAG = "training"
class Run :

    def __init__(self, network, dataset, run_label) :
        self.run_label = run_label
        self.dataset = dataset
        self.network = network

        self.learning_rate = None
        self.batch_size = None
        self.testing_batch_size = None
        self.logging = False
        self.saving = False
        self.iter_sum = 0
        self.restoring = False

    #TODO refractor so we do not need to different object for train/test
    def initialize(self) :
        graph = tf.Graph()
        self.train_prep = Preprocessor(self.dataset, self.batch_size, graph, training = True)
        self.test_prep = Preprocessor(self.dataset, self.batch_size, graph, training = False)
        self.network.preprocess(self.train_prep)
        self.network.preprocess(self.test_prep)
        self.train_prep.finalize()
        self.test_prep.finalize()

        temp_network = Network(self.network.network_name, graph, (self.train_prep.dequeue_batch, self.test_prep.dequeue_batch))

        self.network.construct_network(temp_network)
        self.network = temp_network

        if not self.network.finalized :
            self.network.finalize()

    def _write_iteration(self) :
        if(self.saving) :
            json.dump({"iter_sum": self.iter_sum}, open(self.meta_path, 'w'))

    def restore_if_possible(self) :
        if(self.saving) :
            if(os.path.exists(self.meta_path)) :
                self.iter_sum = json.load(open(self.meta_path))["iter_sum"]
                self.restoring = True

    def _feed_dict(self, train=True, lr_ratio = 1.0) :
        result = {}

        if(train) :
            result[self.network.learning_rate] = self.learning_rate * lr_ratio

            result[self.network.extra[TRAINING_FLAG]] = True

            for tensor, value in self.network.extra[DROPOUT_NAME] :
                result[tensor] = value

            for tensor in self.network.extra[AUGMENT_NAME] :
                result[tensor] = True

        else :
            result[self.network.extra[TRAINING_FLAG]] = False

            for tensor, value in self.network.extra[DROPOUT_NAME] :
                result[tensor] = 1.0

            for tensor in self.network.extra[AUGMENT_NAME] :
                result[tensor] = False

        return result

    def _get_hyperid(self) :
        return "lr{}bs{}".format(self.learning_rate, self.batch_size)

    def _get_runid(self) :
        return "{}{}".format(self.run_label, self._get_hyperid())

    #TODO make clear that enable logging must be ran after hyperparameters are set
    def enable_logging(self, log_dir) :
        self.network.enable_tensorboard()
        self.train_writer = tf.summary.FileWriter(log_dir + "/" + self.network.network_name + "/" + self._get_runid() + "/train", self.network.graph)
        self.test_writer = tf.summary.FileWriter(log_dir + "/" + self.network.network_name + "/" + self._get_runid() + "/test")
        self.log_dir = log_dir
        self.logging = True

    def enable_saving(self, save_dir, save_freq=1, keep_recent=1, keep_n_hours=10000) :
        save_path = os.path.join(save_dir, self.network.network_name, self.run_label, self._get_hyperid())
        os.makedirs(save_path, exist_ok=True)
        self.meta_path = os.path.join(save_path, "info.json")
        self.save_path = os.path.join(save_path, "save")
        self.network.enable_saving(keep_recent, keep_n_hours)
        self.keep_recent = keep_recent
        self.keep_n_hours = keep_n_hours
        self.save_freq=save_freq
        self.saving = True

    def _prep_session(self) :
        self.session = tf.Session(graph=self.network.graph)
        self.session.run(self.network.initialize)
        if self.testing_batch_size == None :
            self.testing_batch_size = self.batch_size

    def _run_test(self, final=False) :
        run_acc = 0
        run_loss = 0

        while True :

            try : 
                current_loss, current_accuracy = self.session.run([self.network.cost_function, self.network.accuracy], feed_dict=self._feed_dict(train=False))

                run_acc += (current_accuracy  * self.batch_size)
                run_loss += (current_loss  * self.batch_size)

            except tf.errors.OutOfRangeError :
                break

        if (not final) :
            self.test_prep.start(1, self.session, feed_threads=8)

        return ((run_acc / self.dataset.testing_len()), (run_loss / self.dataset.testing_len()))
    

    def run(self, epochs, verbose=False) :
        self._prep_session()

        set_size = self.dataset.training_len()
        iter_per_epoch = math.floor(set_size / self.batch_size)
        print("iter_per_epoc : {}".format(iter_per_epoch))

        if (verbose) :
            print("starting run")

        if (self.restoring) :
            print("restoring previous run")
            self.network.saver.restore(self.session, self.save_path + "-" + str(self.iter_sum))

        self.train_prep.start(epochs, self.session, feed_threads=16)
        self.test_prep.start(1, self.session, feed_threads=8)

        looping = True
        while looping :

            try :
                self.iter_sum += 1
                print("iter {}".format(self.iter_sum))

                if (self.logging and (self.iter_sum % 10 == 0)) :
                    _, summary =self. session.run([self.network.train_step, self.network.summaries], feed_dict=self._feed_dict(train=True))
                    self.train_writer.add_summary(summary, self.iter_sum)
                else :
                    self.session.run(self.network.train_step, feed_dict=self._feed_dict(train=True))
            except tf.errors.OutOfRangeError :
                print("completed training")
                looping = False

            if (self.iter_sum % iter_per_epoch == 0) :
                print("starting test")
                e = self.iter_sum / iter_per_epoch
                current_accuracy, current_loss = self._run_test(self.dataset)

                if verbose :
                    print("Epoch: {}; Acc: {}; Loss: {}".format(e + 1, current_accuracy, current_loss))

                if self.logging :
                    test_summ = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy", simple_value=current_accuracy),
                        tf.Summary.Value(tag="loss", simple_value=current_loss)
                        ])
                    self.test_writer.add_summary(test_summ, self.iter_sum)

                if self.saving and (e % self.save_freq == 0) :
                    print("saving epoch")
                    print(self.save_path)
                    self.network.saver.save(self.session, self.save_path, global_step=self.iter_sum)
                    self._write_iteration()


        if(self.logging) : 
            self.test_writer.close()
            self.train_writer.close()

class Network :

#internals
############################################################################   

    def __init__(self, name, graph, dequeue_ops) :
        self.network_name = name
        self.graph = graph
        self.dequeue_ops = dequeue_ops
        self.layers = []
        self.names = []
        self.weights = {}
        self.biases = {}
        self.extra = {DROPOUT_NAME : [], AUGMENT_NAME : []}
        self.tboard = False
        self.finalized = False
        self._setup_input_layer()

    def _next_layer(self) :
        self.layers.append([])
        self.names.append([])
        self.current_layer = len(self.layers) - 1

    def _add(self, name, op) :
        self.layers[self.current_layer].append(op)
        self.names[self.current_layer].append(name)

    def _create_weight(self, shape, name) :
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.weights[name] = w
        return w

    def _create_bias(self, shape, name) :
        b = tf.Variable(tf.constant(0.1, shape=shape))
        self.biases[name] = b
        return b

    def _gen_name(self, type) :
        return type + str(self.current_layer)

    def _get_previous_tensor(self) :
        result = None

        if (len(self.layers[-1]) == 0) :
            result = self.layers[-2][-1]
        else :
            result = self.layers[-1][-1]

        return result

    def _get_previous_tensor_name(self) :
        result = None

        if (len(self.names[-1]) == 0) :
            result = self.names[-2][-1]
        else :
            result = self.names[-1][-1]

        return result

    def get_labels(self) :
        return self.layers[0][0]

    def get_input(self) :
        return self.layers[0][1]

    def get_dropouts(self) :
        return self.dropouts

    def _use_cross_entropy(self) :
        self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.get_labels() * tf.log(self._get_previous_tensor() + 1e-10), reduction_indices=[1]))

    def _setup_input_layer(self) : #, data_shape, label_shape) :
        self._next_layer()
        with self.graph.as_default(), self.graph.name_scope(self._gen_name(INPUT_NAME)) :
            training_flag = tf.placeholder(tf.bool)
            self.extra[TRAINING_FLAG] = training_flag
            inp, lab = tf.cond(training_flag, lambda:self.dequeue_ops[0], lambda:self.dequeue_ops[1])
            self._add(self._gen_name(LABEL_NAME), lab)
            self._add(self._gen_name(INPUT_NAME), inp)

#learning layers
############################################################################
    def add_conv_layer(self, filter_size, filter_depth, stride=1, activation=tf.nn.relu) :
        self._next_layer()
        name = self._gen_name(CONV_NAME)
        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()
            oldshape = inp.get_shape().dims
            weight = self._create_weight([filter_size, filter_size, oldshape[-1].value, filter_depth], name)
            bias = self._create_bias([filter_depth], name)
            conv = activation(tf.nn.conv2d(inp, weight, strides=[1, stride, stride, 1], padding = 'SAME') + bias)
            #TODO keep track of pre and post activation?
            self._add(name, conv)

    def add_max_pool(self, kernel_size, stride) :
        name = self._gen_name(POOL_NAME)
        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()
            pool = tf.nn.max_pool(inp, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME')
            self._add(name, pool)

    def add_lr_normalization(self, depth=5, bias=1, alpha=1, beta=0.5) :
        name = self._gen_name(LOCAL_NORM)
        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()
            norm = tf.nn.local_response_normalization(inp, depth_radius=depth, bias=bias, alpha=alpha, beta=beta)
            self._add(name, norm)

    def add_dropout(self, keep_prob) :
        name = self._gen_name(DROPOUT_NAME)

        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()
            keep = tf.placeholder(tf.float32)
            self.extra[DROPOUT_NAME].append((keep, keep_prob))
            drop = tf.nn.dropout(inp, keep)
            self._add(name, drop)

    def add_full_layer(self, depth, activation=tf.nn.relu) :
        #TODO only reshape if necessary
        self._next_layer()
        name = self._gen_name(FULL_NAME)
        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()
            old_dims = inp.get_shape().dims
            flat_shape = 1
            for d in old_dims :
                if (d.value != None) :
                    flat_shape *= d.value

            weights = self._create_weight([flat_shape, depth], name)
            biases = self._create_bias([depth], name)
            flat_input = tf.reshape(inp, [-1, flat_shape])
            full = activation(tf.matmul(flat_input, weights) + biases)
            self._add(name, full)

#optimizers
############################################################################
    def use_SGD(self) :
        with self.graph.as_default() :
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    def use_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-08) :
        with self.graph.as_default() :
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
            
    def enable_tensorboard(self) :
        self.tboard = True
        with self.graph.as_default() :
            tf.summary.scalar("loss", self.cost_function)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summaries = tf.summary.merge_all()

    def enable_saving(self, max_to_keep, keep_n_hours) :
        with self.graph.as_default() :
            self.saver = tf.train.Saver(max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_n_hours)

    def finalize(self) :
        with self.graph.as_default() :
            self._use_cross_entropy()
            self.train_step = self.optimizer.minimize(self.cost_function)
            self.prediction = tf.argmax(self._get_previous_tensor(), 1)
            self.check_prediction = tf.equal(tf.argmax(self._get_previous_tensor(), 1), tf.argmax(self.get_labels(), 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.check_prediction, tf.float32))
            self.initialize = tf.global_variables_initializer()
            self.finalized = True


