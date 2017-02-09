import tensorflow as tf
import numpy as np

NETWORK_STAGE = "network"
PREPROCESS_STAGE = "preprocess"
LOSS_STAGE = "loss"
ACCURACY_STAGE = "accuracy"
OPTIMIZER_STAGE = "optimizer"

class Layer :

    tag = "layertag"
    stage = "layertype"

    def __init__(self) :
        raise NotImplementedError()

    def build_layer(self, util) :
        raise NotImplementedError()

    def build_eval_layer(self, util) :
        return self.build_layer(util) #only overwrite if you need it to be overwritten

    def parameter_spec(self) :
        return ()#only overwrite if you need it to be overwritten


class FullLayer(Layer) :
    tag = "full"
    stage = NETWORK_STAGE

    def __init__(self, depth, activation="relu") :
        self.depth = depth
        self.activation=activation

    def build_layer(self, layer_util) :
        act = layer_util.get_activation(self.activation)
        inp = layer_util.get_previous_tensor()
        old_dims = inp.get_shape().dims

        flat_shape = 1

        for d in old_dims :
            if (d.value != None) :
                flat_shape *= d.value

        weight = layer_util.get_weight([flat_shape, self.depth])
        bias = layer_util.get_bias([self.depth])
        flat_input = tf.reshape(inp, [-1, flat_shape])
        result = act(tf.matmul(flat_input, weight) + bias)

        return result

    def parameter_spec(self) :
        return (("depth", self.depth),("activation", self.activation))

class SoftmaxLossLayer(Layer) :
    tag="softmax_loss"
    stage = LOSS_STAGE

    def __init__(self) :
        pass

    def build_layer(self, layer_util) :
        pred = layer_util.get_previous_tensor()
        label = layer_util.get_label_tensor()

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label))

class OnehotAccuracyLayer(Layer) :
    tag="accuracy"
    stage = ACCURACY_STAGE

    def __init__(self) :
        pass

    def build_layer(self, layer_util) :
        pred = layer_util.get_previous_tensor()
        label = layer_util.get_label_tensor()

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32))

class GradDescentLayer(Layer) :
    tag = "gradient_descent"
    stage = OPTIMIZER_STAGE

    def __init__(self) :
        pass

    def build_layer(self,layer_util) :
        loss = layer_util.get_previous_tensor()
        learning_rate = layer_util.get_learning_rate()
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    def parameter_spec(self) :
        return (("learning_rate", self.learning_rate),)

