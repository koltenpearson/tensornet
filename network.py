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
    def _get_labels(self) :
        return self.layers[0][0]

    def _get_input(self) :
        return self.layers[0][1]


    def _use_cross_entropy(self) :
        self.cost_function = tf.reduce_mean(-tf.reduce_sum(self._get_labels() * tf.log(self._get_previous_tensor() + 1e-10), reduction_indices=[1]))

    def _write_log(self, string, verbose=False) :
        log = open(self.network_name + ".log", 'a')
        log.write(string + '\n')
        print(string) if verbose else None
        log.close()

    def _write_seperation(self) :
        div = "################################################################################"

        self._write_log('\n')
        self._write_log('\n')
        self._write_log(div)
        self._write_log('\n')


#Visualization functions
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

        print(output[:-4])

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

#runtime 
############################################################################

    def finalize(self, learning_rate) :
        with self.graph.as_default(), self.graph.name_scope("final_setup") :
            self._use_cross_entropy()
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost_function)
            self.learning_rate = learning_rate #keep for logging purposes
            self.check_prediction = tf.equal(tf.argmax(self._get_previous_tensor(), 1), tf.argmax(self._get_labels(), 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.check_prediction, tf.float32))
            self.initialize = tf.initialize_all_variables()


    def _run_test(self, batch_size, test_data, test_labels, session) :
        run_acc = 0
        run_loss = 0

        input_holder = self._get_input()
        label_holder = self._get_labels()

        iterations = math.ceil(test_data.shape[0] / batch_size)

        for i in range(iterations) :
            batch_start = (i * batch_size)
            batch_end = batch_start + batch_size
            if (batch_end > test_data.shape[0]) :
                batch_end = test_data.shape[0]

            current_accuracy, current_loss = session.run([self.accuracy, self.cost_function],
                                           feed_dict = {
                                               input_holder:test_data[batch_start:batch_end],
                                               label_holder:test_labels[batch_start:batch_end],
                                               })
            run_acc += (current_accuracy  * (batch_end - batch_start))
            run_loss += (current_loss  * (batch_end - batch_start))

        return ((run_acc / test_data.shape[0]), (run_loss / test_data.shape[0]))
    
    #TODO  vairable test batch size here 
    def run_network(self, epochs, batch_size, data, labels, test_data, test_labels, verbose=False) :
        iterations = math.ceil(data.shape[0] / batch_size)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)

        self._write_seperation()
        message = "STARTING RUN epochs: {} batch_size: {} ({} iterations per epoch) learning_rate: {}".format(epochs, batch_size, iterations, self.learning_rate)
        date = datetime.datetime.today()
        message += "\ndate:{}".format(str(date))
        self._write_log(message, verbose=verbose)

        input_holder = self._get_input()
        label_holder = self._get_labels()

        for e in range(epochs) :

            for i in range(iterations) :

                batch_start = (i * batch_size) 
                batch_end = batch_start + batch_size
                if (batch_end > data.shape[0]) :
                    batch_end = data.shape[0]
                    self._write_log("WARNING: clipping batch", verbose=verbose)


                if (i % 100 == 0) :
                    acc = session.run(self.accuracy, 
                            feed_dict={
                                input_holder:data[batch_start:batch_end],
                                label_holder:labels[batch_start:batch_end],
                                })
                    #print("iter: {} acc: {}".format(i, acc))

                session.run(self.train_step, 
                            feed_dict={
                                input_holder:data[batch_start:batch_end],
                                label_holder:labels[batch_start:batch_end],
                                })

            current_accuracy, current_loss = self._run_test(batch_size, test_data, test_labels, session)

            self._write_log("Epoch: {}; Acc: {}; Loss: {}".format(e + 1, current_accuracy, current_loss), verbose = verbose)

        fdate = datetime.datetime.today()
        self._write_log("Finish Date : {} ".format(fdate), verbose = verbose)
        etime = fdate - date
        self._write_log("Elapsed Time : {} days {} hours {} minutes ".format(etime.days, etime.seconds // (60**2), (etime.seconds % (60**2)) // 60), verbose=verbose)

