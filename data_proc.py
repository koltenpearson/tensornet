#Some functions to aid in data processing

import numpy as np
import idx2numpy
import os

def get_category_map(source) :
    result = {}
    count = 0

    for s in source :
        
        if (s not in result.keys()) :
            result[s] = count
            count += 1

    return result

#right now uses dtype of uint8, giving an upper limit of 256 labels
def to_one_hot(source, cat_map) :
    categories = cat_map
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

    train_images = train_images.reshape([-1, 28, 28, 1])/256
    test_images = test_images.reshape([-1, 28, 28, 1])/256
    cat_map = get_category_map(test_labels)
    train_labels = to_one_hot(train_labels, cat_map)
    test_labels = to_one_hot(test_labels, cat_map)

    return (train_images, train_labels, test_images, test_labels)

def validate_mnist() :
    import matplotlib.pyplot as pp
    ti, tl, testi, testl = get_mnist()

    for i,j in zip(ti,tl) :
        print(j)
        pp.imshow(i.reshape([28,28]), cmap='Greys')
        pp.show()
