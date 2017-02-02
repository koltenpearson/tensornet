import tensorflow as tf
import numpy as np
import threading
import math


class Preprocessor :

    def __init__(self, prep, data, label) :
        self.data = data
        self.label = label
        self.coord = tf.train.Coordinator()
        self.prep = prep

        self.datapoints = len(self.data)

    def _index_thread(self, epochs, session, shuffle=False) :
        print("starting indexing thread")
        for e in range(epochs) :
            print("starting epoch with {} images".format(self.datapoints))
            indices = np.arange(self.datapoints)
            if (shuffle) :
                np.random.shuffle(indices)

            for i in indices :
                #print("feeding {}".format(i))
                session.run(self.prep.index_enqueue, feed_dict={self.prep.index:i})

                if self.coord.should_stop() :
                    print('returning')
                    return

        print('closing index queue and exiting')
        session.run(self.prep.index_queue.close())

    def _feeder_thread(self, session) :
        print("starting feeder thread")

        while not self.coord.should_stop() :

            try :
                index = session.run(self.prep.index_dequeue)
                session.run(self.prep.batch_enqueue, feed_dict={
                                                        self.prep.image_in : self.data[index], 
                                                        self.prep.label_in : self.label[index]
                                                     })

            except tf.errors.OutOfRangeError :
                print("queue is closed and empty, exiting")
                break

    def _closer_thread(self, session) :
        self.coord.join()
        print("all threads finished, closing queue")
        session.run(self.prep.batch_queue.close())


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
