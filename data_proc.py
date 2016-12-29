#Some functions to aid in data processing

import numpy as np
import idx2numpy
import h5py
import os

def get_category_list(source) :
    result_list = []

    for s in source :
        if (s not in result_list) :
            result_list.append(s)

    return result_list

def to_one_hot(source, cat_list) :

    categories = {}
    for i, c in enumerate(cat_list) :
        categories[c] = i

    dims = source.shape
    result = np.zeros((dims[0], len(categories.keys())), dtype=np.uint8)
    
    for i, s in enumerate(source) :

        result[i][categories[s]] = 1

    return result

def get_norms(dataset) :

    max = np.amax(dataset)
    min = np.amin(dataset)

    return (min, max)

def get_mean(dataset) :
    return np.mean(dataset, dtype=np.float64)

def shuffle_data(images, labels) :
    indices = np.shuffle(np.arange(len(images)))

    return images[indices], labels[indices]


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

    train_images = train_images.reshape([-1, 28, 28, 1])
    test_images = test_images.reshape([-1, 28, 28, 1])
    cat_list = get_category_list(test_labels)
    train_labels = to_one_hot(train_labels, cat_list)
    test_labels = to_one_hot(test_labels, cat_list)

    return (train_images, train_labels, test_images, test_labels)

def validate_mnist() :
    import matplotlib.pyplot as pp
    ti, tl, testi, testl = get_mnist()

    for i,j in zip(ti,tl) :
        print(j)
        pp.imshow(i.reshape([28,28]), cmap='Greys')
        pp.show()

def make_mnist_h5py(loc) :
    dat = h5py.File(loc, 'w')
    train_image, train_label, test_image, test_label = get_mnist()
    cat_list = get_category_list(test_labels)
    min, max = get_norms(train_image)
    mean = get_mean(train_image)

    cat_list = [str(l) for l in cat_list]

    out_train_image = dat.create_dataset('train_image', data=train_image, dtype=np.uint8)
    out_train_label = dat.create_dataset('train_label',  data=train_label, dtype=np.uint8)
    out_test_image = dat.create_dataset('test_image', data=test_image, dtype=np.uint8)
    out_test_label = dat.create_dataset('test_label', data=test_label, dtype=np.uint8)
    out_label_info = dat.create_dataset('label_info', data=cat_list, dtype=h5py.special_dtype(vlen=unicode))
    out_max = dat.create_dataset('max', data=max, dtype=np.float64)
    out_max = dat.create_dataset('min', data=min, dtype=np.float64)
    out_max = dat.create_dataset('mean', data=mean, dtype=np.float64)
