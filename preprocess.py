import tensorflow as tf
import numpy as np
import threading
import math

PREPROCESS_NAME = "preprocess"

MIN = "min"
MAX = "max"
MEAN = "mean"
STDDEV = "stddev"


class Preprocessor :

    def __init__(self, dataset, batch_size, graph, training=True) :
        self.dataset = dataset
        self.graph = graph
        self.batch_size = batch_size
        self.coord = tf.train.Coordinator()
        self.sem = threading.Semaphore(0)
        self.tensors = []
        self.training=training

        if training :
            self.data, self.label = self.dataset.get_training_data() 
        else :
            self.data, self.label = self.dataset.get_testing_data() 

        self.datapoints = len(self.data)

        print("initializing preprocessor")
        with self.graph.as_default() :
            self.incoming_index = tf.placeholder(tf.int32, shape=())
            self.training_flag = tf.placeholder(tf.bool)

            self.index_queue = tf.FIFOQueue(self.batch_size * 5, dtypes=[tf.int32])
            self.index_dequeue = self.index_queue.dequeue()
            self.index_enqueue = self.index_queue.enqueue(self.incoming_index)
            self._setup_initial_tensor()

    def _setup_initial_tensor(self) :
        shape = self.data.shape
        self.image = tf.placeholder(tf.float32, shape=shape[1:])
        self.tensors.append(self.image)

    def _get_previous_tensor(self) :
        return self.tensors[-1]

    def _add(self, tensor) :
        self.tensors.append(tensor)

    def _index_thread(self, epochs, session) :
        print("starting indexing thread")
        for e in range(epochs) :
            print("starting epoch with {} images".format(self.datapoints))
            indices = np.arange(self.datapoints)
            #if (self.training) :
            #    np.random.shuffle(indices)
            for i in indices :
                if(not self.training) :
                    print("feeding {}".format(i))
                session.run(self.index_enqueue, feed_dict={self.incoming_index:i})

                if self.coord.should_stop() :
                    print('returning')
                    return

        print('closing index queue and exiting')
        session.run(self.index_queue.close())

    def _feeder_thread(self, session) :
        print("starting feeder thread")

        while not self.coord.should_stop() :

            try :
                index = session.run(self.index_dequeue)
                session.run(self.preprocess, feed_dict={self.image:self.data[index], self.label_holder:self.label[index], self.training_flag:self.training})
            except tf.errors.OutOfRangeError :
                print("queue is closed and empty, exiting")
                break

    def _closer_thread(self, session) :
        self.coord.join()
        print("all threads finished, closing queue")
        session.run(self.batch_queue.close())


    def finalize(self) :
        print("finalizing preprocessor")
        with self.graph.as_default() :
            inp = self._get_previous_tensor()

            shape = inp.get_shape().dims
            label_shape = self.label.shape[1:]
            self.label_holder = tf.placeholder(tf.float32, shape=label_shape)

            self.batch_queue = tf.FIFOQueue(self.batch_size * 5, [tf.float32, tf.float32], [shape, label_shape])

            self.preprocess = self.batch_queue.enqueue((inp, self.label_holder))
            self.dequeue_batch = self.batch_queue.dequeue_up_to(self.batch_size)

    def start(self, epochs, session, feed_threads=16) :
        t = threading.Thread(target=self._index_thread, args=(epochs, session))
        t.daemon = True
        self.coord.register_thread(t)
        t.start()

        for i in range(feed_threads) :
            t = threading.Thread(target=self._feeder_thread, args=(session,))
            t.daemon = True
            self.coord.register_thread(t)
            t.start()

        t = threading.Thread(target=self._closer_thread, args=(session,))
        t.daemon = True
        t.start()

#preprocessing layers
############################################################################

    def add_standard_norm(self) :
        name = PREPROCESS_NAME
        with self.graph.as_default(), self.graph.name_scope(name) :
            mean = self.dataset.get_mean()
            stddev = self.dataset.get_stddev()
            inp = self._get_previous_tensor()
            out = (inp - mean) / stddev
            self._add(out)

    def add_minmax_norm(self) :
        print("adding minmax norm")
        name = PREPROCESS_NAME
        with self.graph.as_default(), self.graph.name_scope(name) :
            minval = self.dataset.get_min()
            maxval = self.dataset.get_max()
            inp = self._get_previous_tensor()
            out = (inp - minval) / (maxval -  minval)
            self._add(out)

    def add_random_crop(self, target_size, test_type="resize") :
        #NOTE assumes square inputs 
        #Also assumse that the target sizes divides evenly into 2
        name = PREPROCESS_NAME
        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()

            shape = inp.get_shape().dims
            offset = math.floor((shape[1].value - target_size)/2)

            #NOTE: tightly coupled to ordering Height x Width x Depth
            true = lambda : tf.random_crop(inp, (target_size, target_size, shape[2].value))
            false = lambda : tf.image.resize_images(inp, (target_size, target_size), method=tf.image.ResizeMethod.BICUBIC)
            if test_type == "center" :
                false = lambda : tf.image.crop_to_bounding_box(inp, offset, offset, target_size, target_size)
            #TODO center crop or resize?

            out = tf.cond(self.training_flag, true, false)
            self._add(out)

    def add_random_flip(self) :
        name = PREPROCESS_NAME
        with self.graph.as_default(), self.graph.name_scope(name) :
            inp = self._get_previous_tensor()

            true = lambda : tf.image.random_flip_left_right(inp)
            false = lambda : inp

            out = tf.cond(self.training_flag, true, false)
            self._add(out)

    #TODO add other image distortion


def test_preprocess() :
    import matplotlib.pyplot as pp
    import data_proc

    dataset = data_proc.Dataset("test/dataset/mnist.h5")

    ppr = Preprocessor(dataset, 10, training=True)
    ppr.add_random_crop(25, test_type="center")
    ppr.finalize()

    sess = tf.Session(graph = ppr.graph)
    ppr.start(1, sess, feed_threads=6)

    x = sess.run(ppr.dequeue_batch)
    print(x[0].shape, x[1].shape)
    for i in x[0] :
        pp.imshow(i.reshape((25,25)), cmap="Greys")
        pp.show()

if __name__ == "__main__" :
    test_preprocess()
