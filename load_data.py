import tensorflow as tf
import numpy as np

TRAIN_SET = 'data/abalone_train.csv'
VALIDATION_SET = 'data/abalone_test.csv'
TEST_SET = 'data/abalone_predict.csv'


def load_data():
    # Training examples
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TRAIN_SET, target_dtype=np.int, features_dtype=np.float64)

    # Test examples
    validation_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=VALIDATION_SET, target_dtype=np.int, features_dtype=np.float64)

    # Set of 7 examples for which to predict abalone ages
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=TEST_SET, target_dtype=np.int, features_dtype=np.float64)

    return training_set, validation_set, test_set
