#Some functions to aid in data processing

import numpy as np
import idx2numpy
import os

def _get_category_map(source) :
    result = {}
    count = 0

    for s in source :
        
        if (s not in result.keys()) :
            result[s] = count
            count += 1

    return result

#right now uses dtype of uint8, giving an upper limit of 256 labels
def to_one_hot(source) :
    categories = _get_category_map(source)
    dims = source.shape
    result = np.zeros((dims[0], len(categories.keys())), dtype=np.uint8)
    
    for i, s in enumerate(source) :

        result[i][categories[s]] = 1

    return result


def get_mnist() :
    ROOT = "mnist"
    train_images = os.path.join(ROOT, "train-images.idx")
    train_labels = os.path.join(ROOT, "train-labels.idx")
    test_images = os.path.join(ROOT, "test-images.idx")
    test_labels = os.path.join(ROOT, "test-labels.idx")

    train_images = idx2numpy.convert_from_file(train_images)
    train_labels = idx2numpy.convert_from_file(train_labels)
    test_images = idx2numpy.convert_from_file(test_images)
    test_labels = idx2numpy.convert_from_file(test_labels)

    train_labels = to_one_hot(train_labels)
    test_labels = to_one_hot(test_labels)

    return (train_images, train_labels, test_images, test_labels)
