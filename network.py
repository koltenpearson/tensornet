import tensorflow as tf
import numpy as np
import math
import types
import datetime
import os
import json
from collections import defaultdict

import layer #TODO switch to in package ref


POOL_NAME = "pool"
CONV_NAME = "conv"
FULL_NAME = "fc"
INPUT_NAME = "input"
LABEL_NAME = "label"
LOCAL_NORM = "lrnorm"
DROPOUT_NAME = "dropout"

##This class is used to make it easier to build a network with layers
# it only has a few methods such a person would care about
# the network builder needs to be more aware of its other methods.
#
class LayerUtil :

    def __init__(self, input_tensor, label_tensor) :
        self.activations = {"relu" : tf.nn.relu,"softmax" : tf.nn.softmax, "none" : lambda x:x}
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.random_normal_initializer()
        self.prev_tensor = input_tensor
        self.label_tensor = label_tensor

    #this method will return the last layers output
    # as a tensor
    def get_previous_tensor(self) :
        return self.prev_tensor

    def get_label_tensor(self) :
        return self.label_tensor

    def update_latest(self, tensor) :
        self.prev_tensor = tensor

    #returns a new weight (or existing weight, if it has already been initialized)
    def get_weight(self, shape, dtype=tf.float32) :
        name = "weight"
        try :
            result = tf.get_variable(name, shape, dtype=dtype, initializer=self.weight_initializer)
        except ValueError :
            tf.get_variable_scope().reuse_variables()
            result = tf.get_variable(name, shape, dtype=dtype, initializer=self.weight_initializer)

        return result

    #returns a new bias (or existing if it has already been initialized)
    def get_bias(self, shape, dtype=tf.float32) :
        name = "bias"
        try :
            result = tf.get_variable(name, shape, dtype=dtype, initializer=self.bias_initializer)
        except ValueError :
            tf.get_variable_scope().reuse_variables()
            result = tf.get_variable(name, shape, dtype=dtype, initializer=self.bias_initializer)

        return result


    #returns activation functions based off a key, used to make them serializable
    def get_activation(self, key) :
        return self.activations[key]

class PreProcUtil :

    def __init__(self, dataset, start_tensor) :
        self.stddev = dataset.get_stddev()
        self.mean = dataset.get_mean()
        self.data_min = dataset.get_min()
        self.data_max = dataset.get_max()
        self.prev_tensor = start_tensor

    def get_previous_tensor(self) :
        return self.prev_tensor

    def update_tensor(self, tensor) :
        self.prev_tensor = tensor


#Intermediate class that comes from what the user defined, can be used to generate graph, or other ops
class NetSpec : 

    def __init__(self, params) :
        self.params = defaultdict (lambda : [])
        for p in params :
            self.params[p.stage].append(p)

    def create_network(self, dataset, batch_size) :
        net = Network()
        with net.graph.as_default() :

            self._setup_index_read(net.train_prep, batch_size)
            self._setup_preprocessing(net.train_prep, dataset, batch_size, train=True)
            self._setup_network_graph(net.train, net.train_prep, train=True)
            self._setup_accuracy(net.train, net.train_prep, train=True)
            self._setup_loss(net.train, net.train_prep, train=True)
            self._setup_optimizer(net.train, net.train_prep, train=True)


            self._setup_index_read(net.eval_prep, batch_size)
            self._setup_preprocessing(net.eval_prep, dataset, batch_size, train=False)
            self._setup_network_graph(net.eval, net.eval_prep, train=False)
            self._setup_accuracy(net.eval, net.eval_prep, train=False)
            self._setup_loss(net.eval, net.eval_prep, train=False)

            net.initialize = tf.global_variables_initializer()

        return net

    def _setup_index_read(self, prep, batch_size)  :
        with tf.variable_scope("prep") :
            prep.index = tf.placeholder(tf.int32, shape=())
            index_queue = tf.FIFOQueue(batch_size * 5, dtypes=[tf.int32])
            prep.index_enqueue = index_queue.enqueue(prep.index)
            prep.index_dequeue = index_queue.dequeue()
            prep.index_queue = index_queue

    def _setup_preprocessing(self, prep, dataset, batch_size, train=True) :
        with tf.variable_scope("prep") :
            image_shape, label_shape = dataset.get_shape() #TODO should I even assume these are normalized sizes coming in?
            #TODO do I need the whole dataset for this . . .

            prep.image_in = tf.placeholder(tf.float32, shape=image_shape)
            prep.label_in = tf.placeholder(tf.float32, shape=label_shape)

            p_util = PreProcUtil(dataset, prep.image_in)

            for l in self.params[layer.PREPROCESS_STAGE] :
                if train :
                    tensor = l.build_layer(p_util)
                else :
                    tensor = l.build_eval_layer(p_util)
                p_util.update_tensor(tensor)

            ##now we build the output queues
            inp = p_util.get_previous_tensor()
            batch_queue = tf.FIFOQueue(batch_size * 5, [tf.float32, tf.float32], [inp.get_shape().dims, prep.label_in.get_shape().dims])
            prep.batch_enqueue = batch_queue.enqueue((inp, prep.label_in))
            prep.image_out, prep.label_out = batch_queue.dequeue_up_to(batch_size)
            prep.batch_queue = batch_queue

    def _setup_network_graph(self, net, prep, train=True) :
        l_util = LayerUtil(prep.image_out, prep.label_out)

        for i,l in enumerate(self.params[layer.NETWORK_STAGE]) :
            name = l.tag + str(i)
            with tf.variable_scope(name) :
                if train :
                    tensor = l.build_layer(l_util)
                else :
                    tensor = l.build_eval_layer(l_util)
                l_util.update_latest(tensor)

        net.final = l_util.get_previous_tensor()


    def _setup_accuracy(self, net, prep, train=True) :
        with tf.variable_scope("accuracy") :

            acc = self.params[layer.ACCURACY_STAGE][0]
            l_util = LayerUtil(net.final, prep.label_out)
            if train :
                net.accuracy = acc.build_layer(l_util)
            else :
                net.accuracy = acc.build_eval_layer(l_util)

    def _setup_loss(self, net, prep, train=True) :
        with tf.variable_scope("loss") :

            loss = self.params[layer.LOSS_STAGE][0]
            l_util = LayerUtil(net.final, prep.label_out)
            if train :
                net.loss = loss.build_layer(l_util)
            else :
                net.loss = loss.build_eval_layer(l_util)

    def _setup_optimizer(self, net, prep, train=True) :
        with tf.variable_scope("optimizer") :

            opt = self.params[layer.OPTIMIZER_STAGE][0]
            l_util = LayerUtil(net.loss, prep.label_out)
            if train :
                net.optimizer = opt.build_layer(l_util)
            else :
                net.optimizer = opt.build_eval_layer(l_util)

#just a data holding class for network and desired tensor ops
# essentially a complex struct
class Network :

    def __init__(self) :
        self.graph = tf.Graph()
        self.train_prep = types.SimpleNamespace() #to hold data
        self.eval_prep = types.SimpleNamespace()
        self.train = types.SimpleNamespace()
        self.eval = types.SimpleNamespace() #TODO is this a bad name?


################################################################################
################################################################################
################################################################################


class LayerInfo :
    #TODO get rid of?

    def __init__(self, name, shape_weights, shape_biases, shape_output) :
        self.name = name
        self.weight_shape = shape_weights
        self.bias_shape = shape_biases
        self.output_shape = shape_output

class Run :

    def __init__(self, network, run_label) :
        self.run_label = run_label
        if not network.finalized :
            network.finalize()
        self.network = network

        self.learning_rate = None
        self.batch_size = None
        self.testing_batch_size = None
        self.logging = False
        self.saving = False
        self.iter_sum = 0
        self.restoring = False


    def _write_iteration(self) :
        if(self.saving) :
            json.dump({"iter_sum": self.iter_sum}, open(self.meta_path, 'w'))

    def restore_if_possible(self) :
        if(self.saving) :
            if(os.path.exists(self.meta_path)) :
                self.iter_sum = json.load(open(self.meta_path))["iter_sum"]
                self.restoring = True

    def _feed_dict(self, data, labels, train=True, shuffle=True) :
        input_holder = self.network.get_input()
        label_holder = self.network.get_labels()

        batch_size = self.testing_batch_size
        if(train) :
            batch_size = self.batch_size

        iterations = math.ceil(data.shape[0] / batch_size)
        perm = np.arange(iterations)
        if (shuffle and train) :
            np.random.shuffle(perm)
        for i in perm :
            batch_start = (i * batch_size)
            batch_end = batch_start + batch_size
            lr_ratio = 1

            if (batch_end > data.shape[0]) :
                batch_end = data.shape[0]
                lr_ratio = (batch_end - batch_start) / batch_size

            result = {
                        input_holder:data[batch_start:batch_end],
                        label_holder:labels[batch_start:batch_end],
                     }

            if(train) :
                result[self.network.learning_rate] = self.learning_rate * lr_ratio
                for tensor, value in self.network.get_dropouts() :
                    result[tensor] = value
            else :
                for tensor, value in self.network.get_dropouts() :
                    result[tensor] = 1.0


            yield result

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

    def _initialize(self) :
        self.session = tf.Session(graph=self.network.graph)
        self.session.run(self.network.initialize)
        if self.testing_batch_size == None :
            self.testing_batch_size = self.batch_size

    def _run_test(self, test_data, test_labels) :
        run_acc = 0
        run_loss = 0

        for i, feed_dict in enumerate(self._feed_dict(test_data, test_labels, train=False)) :

            current_loss, current_accuracy = self.session.run([self.network.cost_function, self.network.accuracy], feed_dict=feed_dict)

            batch_start = (i * self.testing_batch_size) #TODO this code is repeated from _feed_dict, any way around that?
            batch_end = batch_start + self.testing_batch_size
            if (batch_end > test_data.shape[0]) :
                batch_end = test_data.shape[0]

            run_acc += (current_accuracy  * (batch_end - batch_start))
            run_loss += (current_loss  * (batch_end - batch_start))

        return ((run_acc / test_data.shape[0]), (run_loss / test_data.shape[0]))
    

    def run(self, epochs, train_data, train_labels, test_data, test_labels, verbose=False) :
        self._initialize()

        # TODO eventually incorporate all this metadata somehow . . .
        # self._write_seperation()
        # message = "STARTING RUN epochs: {} batch_size: {} ({} iterations per epoch) learning_rate: {}".format(epochs, batch_size, iterations, self.learning_rate)
        # date = datetime.datetime.today()
        # message += "\ndate:{}".format(str(date))
        # self._write_log(message, verbose=verbose)

        if (verbose) :
            print("starting run")

        if (self.restoring) :
            print("restoring previous run")
            self.network.saver.restore(self.session, self.save_path + "-" + str(self.iter_sum))

        for e in range(epochs) :

            for feed_dict in self._feed_dict(train_data, train_labels) :
                self.iter_sum += 1

                if (self.logging and (self.iter_sum % 10 == 0)) :
                    _, summary =self. session.run([self.network.train_step, self.network.summaries], feed_dict=feed_dict)
                    self.train_writer.add_summary(summary, self.iter_sum)
                else :
                    self.session.run([self.network.train_step, self.network.accuracy], feed_dict=feed_dict)

            current_accuracy, current_loss = self._run_test(test_data, test_labels)
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

        # fdate = datetime.datetime.today()
        # self._write_log("Finish Date : {} ".format(fdate), verbose = verbose)
        # etime = fdate - date
        # self._write_log("Elapsed Time : {} days {} hours {} minutes ".format(etime.days, etime.seconds // (60**2), (etime.seconds % (60**2)) // 60), verbose=verbose)

class Network_old :

#internals
############################################################################   

    def __init__(self, name) :
        self.network_name = name
        self.graph = tf.Graph()
        self.layers = []
        self.names = []
        self.dropouts = []
        self.weights = {}
        self.biases = {}
        self.tboard = False
        self.finalized = False

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


#visualization/log functions
############################################################################

    def get_info(self) :
        result = []
        for layer_group,name_group in zip(self.layers, self.names) :
            result.append([])
            for l,n in zip(layer_group, name_group) :
                weight = ()
                if (n in self.weights) :
                    weight = self.weights[n].get_shape()

                bias = ()
                if (n in self.biases) :
                    bias = self.biases[n].get_shape()

                out_shape = [d.value for d in l.get_shape().dims]

                result[-1].append(LayerInfo(n, weight, bias, out_shape))

        return result
                    

    def print_network(self) :
        divide = '->'
        output = "Network: {}\n".format(self.network_name)
        for layer_group,name_group in zip(self.layers, self.names) :
            for l,n in zip(layer_group, name_group) :
                weight = ()
                if (n in self.weights) :
                    weight = self.weights[n].get_shape()

                bias = ()
                if (n in self.biases) :
                    bias = self.biases[n].get_shape()
                    
                output += "{}: w{}, b{}, s{} {} ".format(n,
                        weight,
                        bias,
                        [d.value for d in l.get_shape().dims], 
                        divide)

            output += '\n'

        return output[:-4]

#layer functions
############################################################################

    def add_input_layer(self, data_shape, label_shape) :
        self._next_layer()
        with self.graph.as_default(), self.graph.name_scope(self._gen_name(INPUT_NAME)) :
            inp = tf.placeholder(tf.float32, data_shape)
            lab = tf.placeholder(tf.float32, label_shape)
            self._add(self._gen_name(LABEL_NAME), lab)
            self._add(self._gen_name(INPUT_NAME), inp)

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
            self.dropouts.append((keep, keep_prob))
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


