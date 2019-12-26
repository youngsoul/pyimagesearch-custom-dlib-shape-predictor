# import the necessary packages
import argparse
import dlib
import time

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
        help="path to trained dlib shape predictor model")
    ap.add_argument("--xml", required=True,
        help="path to input training/testing XML file")
    args = vars(ap.parse_args())

    # compute the error over the supplied data split and display it to
    # our screen
    print("[INFO] evaluating shape predictor...")
    s = time.time()
    mean_average_error = dlib.test_shape_predictor(args["xml"], args["model"])
    e = time.time()
    print(f"[INFO] error: {mean_average_error}")
    print(f"[INFO] evaluation took: {(e-s)} seconds")
