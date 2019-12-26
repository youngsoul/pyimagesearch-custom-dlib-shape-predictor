import multiprocessing
import argparse
import time

import dlib


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", required=True, help="path to input training XML file")
    ap.add_argument("--model", required=True, help="path to serialized dlib shape predictor model")

    args = vars(ap.parse_args())

    # grab the default options for dlibs shape predictor
    dlib_options = dlib.shape_predictor_training_options()

    # define the depth of each regression tree -- there will be a total
    # of 2^tree_depth leaves in each tree; small values of tree_depth
    # will be faster but less accurate while larger valuew will generate
    # trees that are deeper, more accuracte but will run
    # far slower when making predictions
    # typical values range from 2-8
    dlib_options.tree_depth = 2 #4

    # nu - regularization
    # regularization parameter in the range [0,1] that is used to help
    # our model generalize.  Values closer to 1 will make our model fit the
    # training data better, but could cause overfitting.  Values closer to 0
    # will help out model generalize but will require us to have
    # training data in the orde of 1000s of data points
    dlib_options.nu = 0.25 #0.1

    # the number of cascades used to train the shape predictor -- this
    # parameter has a *dramatic* impact on both the accuracy and output size
    # of your model; the more cascades you have, the more accurate
    # your model can potentially be, but also the larger the output size
    # typical: [6,18]
    dlib_options.cascade_depth = 12 #15

    # number of pixels used to generate features for the random trees at each cascade
    # larger pixel values will make your shape predictor more accurate, but slower;
    # use large values if speed is not a problem, otherwise smaller values for resource
    # contrained/embedded devices
    dlib_options.feature_pool_size = 500 #400

    # selects best features at each cascade when training -- the larger this value is
    # the longer it will take to train but (potentially) the more accurate the model
    dlib_options.num_test_splits = 100 #50

    # controls amount of 'jitter' (i.e. data augmentation) when training the shape
    # predictor -- applies the supplied number of random deformations, thereby
    # performing regularization and increasing the ability of our model to generalize
    # typically: [0,50]
    dlib_options.oversampling_amount = 20 #5

    # amount of translation jitter to apply -- the dlib docs recommend
    # values in the range [0, 0.5]
    dlib_options.oversampling_translation_jitter = 0 #0.1

    # tell the dlib shape predictor to be verbose and print out status
    # messages our model trains
    dlib_options.be_verbose = True

    # number of threads/CPU cores to be used when training -- we default
    # this value to the number of available cores on the system, but you
    # can supply an integer value here if you would like
    dlib_options.num_threads = multiprocessing.cpu_count()


    # log our training options to the terminal
    print("[INFO] shape predictor options:")
    print(dlib_options)

    s = time.time()
    # train the shape predictor
    print("[INFO] training shape predictor...")
    dlib.train_shape_predictor(args["training"], args["model"], dlib_options)
    e = time.time()
    print(f"Training took:  {(e-s)} seconds")