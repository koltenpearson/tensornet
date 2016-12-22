import tensorflow as tf
import json
import numpy as np
from pystalkd.Beanstalkd import Connection
from .network import Network

IN_TAG = "_in"
OUT_TAG = "_out"

class FF_listener :

    def __init__(self, server, port, network, checkpoint, name=None) :
        self.network = network
        network.enable_saving(10000,10000)
        self.checkpoint = checkpoint
        self.name = name
        if (name is None) :
            name = self.network.network_name

        self.connection = Connection(server, port)
        self.connection.use(self.name + OUT_TAG)
        self.connection.ignore("default")
        self.connection.watch(self.name + IN_TAG)
        self.connection.ignore("default")



    def listen(self) :
        print("starting server")

        print("initializing network with weights at {}".format(self.checkpoint))
        session = tf.Session(graph=self.network.graph)
        self.network.saver.restore(session, self.checkpoint)


        while(True) :

            job = self.connection.reserve_bytes()
            image = np.frombuffer(job.body, dtype=np.uint8).reshape((1,28,28,1)) #TODO variable sizes in
            #TODO describe normalization steps somewhere and make it variable as well

            normalized_image = (image/255).astype(np.float32)

            feed_dict = {network.data: normalized_image}

            for tensor, _ in self.network.get_dropouts() :
                feed_dict[tensor] = 1.0


            result = session.run(network.prediction, feed_dict = feed_dict)

            message = {"id" : job.jid, "result" : result}

            job.delete()
            self.connection.put(json.dumps(message))
