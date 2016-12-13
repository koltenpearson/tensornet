import tensorflow as tf
import numpy as np
import math
import datetime


POOL_NAME = "pool"
CONV_NAME = "conv"
FULL_NAME = "fc"
INPUT_NAME = "input"
LABEL_NAME = "label"

class LayerInfo :

    def __init__(self, name, shape_weights, shape_biases, shape_output) :
        self.name = name
        self.weight_shape = shape_weights
        self.bias_shape = shape_biases
        self.output_shape = shape_output

class Run :

    def __init__(self, network, run_label="") :
        self.run_label = run_label
        if not network.finalized :
            network.finalize()
        self.network = network

        self.learning_rate = None
        self.batch_size = None
        self.testing_batch_size = None
        self.logging = False


    def _feed_dict(self, data, labels, train=True) :
        input_holder = self.network.get_input()
        label_holder = self.network.get_labels()

        batch_size = self.testing_batch_size
        if(train) :
            batch_size = self.batch_size

        iterations = math.ceil(data.shape[0] / batch_size)
        for i in range(iterations) :
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

            yield result

    def _get_runid(self) :
        return "{}lr{}bs{}".format(self.run_label, self.learning_rate, self.batch_size)

    #TODO make clear that enable logging must be ran after hyperparameters are set
    def enable_logging(self, log_dir) :
        self.network.enable_tensorboard()
        self.train_writer = tf.train.SummaryWriter(log_dir + "/" + self.network.network_name + "/" + self._get_runid() + "/train", self.network.graph)
        self.test_writer = tf.train.SummaryWriter(log_dir + "/" + self.network.network_name + "/" + self._get_runid() + "/test")
        self.logging = True

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

        iter_sum = 0
        for e in range(epochs) :

            if (verbose) :
                print("starting run")


            for feed_dict in self._feed_dict(train_data, train_labels) :
                iter_sum += 1

                if (self.logging and (iter_sum % 10 == 0)) :
                    _, summary =self. session.run([self.network.train_step, self.network.summaries], feed_dict=feed_dict)
                    self.train_writer.add_summary(summary, iter_sum)
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
                self.test_writer.add_summary(test_summ, iter_sum)


        if(self.logging) : 
            self.test_writer.close()
            self.train_writer.close()

        # fdate = datetime.datetime.today()
        # self._write_log("Finish Date : {} ".format(fdate), verbose = verbose)
        # etime = fdate - date
        # self._write_log("Elapsed Time : {} days {} hours {} minutes ".format(etime.days, etime.seconds // (60**2), (etime.seconds % (60**2)) // 60), verbose=verbose)

class Network :

#internals
############################################################################   

    def __init__(self, name) :
        self.network_name = name
        self.graph = tf.Graph()
        self.layers = []
        self.names = []
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

#tensorboard
############################################################################
    def enable_tensorboard(self) :
        self.tboard = True
        with self.graph.as_default() :
            tf.summary.scalar("loss", self.cost_function)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summaries = tf.merge_all_summaries()

    def finalize(self) :
        with self.graph.as_default() :
            self._use_cross_entropy()
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)
            self.check_prediction = tf.equal(tf.argmax(self._get_previous_tensor(), 1), tf.argmax(self.get_labels(), 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.check_prediction, tf.float32))
            self.initialize = tf.initialize_all_variables()
            self.finalized = True


