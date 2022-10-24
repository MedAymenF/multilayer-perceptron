#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from numpy.random import default_rng
from model import MultiLayerPerceptron
import matplotlib.pyplot as plt


def data_splitter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)\
 into a training and a test set,
while respecting the given proportion of examples to be kept\
 in the training set.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset\
 that will be assigned to the
    training set.
Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
Raises:
    This function should not raise any Exception.
"""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    if not isinstance(proportion, (int, float)):
        print('proportion has to be a float.')
        return None
    if proportion < 0 or proportion > 1:
        print('proportion has to be between 0 and 1.')
        return None
    rng = default_rng(1337)
    z = np.hstack((x, y))
    rng.shuffle(z)
    x, y = z[:, :-1].reshape(x.shape), z[:, -1].reshape(y.shape)
    idx = int((x.shape[0] * proportion))
    x_train, x_test = np.split(x, [idx])
    y_train, y_test = np.split(y, [idx])
    return (x_train, x_test, y_train, y_test)


def check_nn_gradients():
    input_layer_size = 3
    hidden_layer_1_size = 4
    hidden_layer_2_size = 4
    num_labels = 3
    m = 5
    lambda_ = 2

    architecture = [input_layer_size, hidden_layer_1_size,
                    hidden_layer_2_size, num_labels]
    mlp = MultiLayerPerceptron(architecture,
                               lambda_=lambda_, output_layer='softmax')

    X = default_rng(1337).uniform(size=(m, input_layer_size))
    y = (np.arange(m) % num_labels).reshape(-1, 1)

    def nn_cost(theta):
        return mlp.nn_cost_function(theta, X, y)

    J, grad = nn_cost(mlp.theta)
    grad = grad.reshape(-1)

    numgrad = mlp.compute_numerical_gradient(nn_cost, mlp.theta)
    diff = np.sqrt(((numgrad - grad) ** 2).sum())\
        / np.sqrt(((numgrad + grad) ** 2).sum())
    print(f'\nThe relative difference between our gradient (analytical)\
 and the numerical gradient is {diff}')
    print(np.hstack([grad.reshape(-1, 1), numgrad.reshape(-1, 1)]))


COLUMNS = [['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se',
            'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
            'concavity_se', 'concave points_se', 'symmetry_se',
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst']]

if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:7.4f}'})

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a Multilayer\
 Perceptron and use it for prediction on new data.')
    parser.add_argument('--dataset', type=str, help='The path to the\
 csv file containing the training/test data.')
    parser.add_argument('--predict', type=str, help='The path to the\
 saved model.')
    args = parser.parse_args()

    if (args.predict):
        # Load model architecture and parameters
        print(f"> loading model '{args.predict}' from disk...")
        with open(args.predict, 'rb') as f:
            minimum = np.load(f)
            rng = np.load(f)
            architecture = np.load(f)
            output_layer = np.load(f)
            lambda_ = np.load(f)
            theta = np.load(f)

        # Read data
        df = pd.read_csv(args.dataset, header=None, index_col=0)
        df.columns = COLUMNS
        df = df[['area_worst', 'smoothness_worst', 'texture_mean',
                 'diagnosis']]
        y = df.pop('diagnosis')
        y = y.replace({'M': 1, 'B': 0})

        # Convert dataframes to numpy
        x_test = df.to_numpy()
        y_test = y.to_numpy().reshape(-1, 1)
        (m, n) = x_test.shape

        # Normalize data
        x_test = (x_test - minimum) / rng

        # Calculate model predictions on the test set
        mlp = MultiLayerPerceptron(architecture, lambda_=lambda_,
                                   init_theta=theta, output_layer=output_layer)
        predictions_test = mlp.predict(x_test, verbose=True, y=y_test)

        # Calculate test set accuracy
        correct = (predictions_test == y_test).sum()
        acc = correct / m
        print(f"\n> correctly predicted ({correct}/{m})")
        print(f'> Test set accuracy = {acc:.4f}')

        # Calculate the binary crossentropy loss
        # and the mean squared error on the test set
        J_test, _ = mlp.nn_cost_function(mlp.theta, x_test, y_test)
        mse_test = mlp.mse(predictions_test, y_test)
        print(f'> loss (binary crossentropy) : {J_test:.4f}')
        print(f'> loss (mean squared error) : {mse_test:.4f}')
    elif (args.dataset):
        # Read data
        df = pd.read_csv(args.dataset, header=None, index_col=0)
        df.columns = COLUMNS
        df = df[['area_worst', 'smoothness_worst', 'texture_mean',
                 'diagnosis']]
        y = df.pop('diagnosis')
        y = y.replace({'M': 1, 'B': 0})

        # Convert dataframes to numpy
        x = df.to_numpy()
        y = y.to_numpy().reshape(-1, 1)

        # Split the dataset into a training set and a validation set
        (x_train, x_valid, y_train, y_valid) = data_splitter(x, y, 0.8)
        (m, n) = x_train.shape
        y_train = y_train.astype('int')
        y_valid = y_valid.astype('int')
        print(f'x_train shape : {x_train.shape}')
        print(f'x_valid shape : {x_valid.shape}')

        # Normalize data
        minimum = x_train.min(axis=0)
        rng = x_train.max(axis=0) - minimum
        x_train = (x_train - minimum) / rng
        x_valid = (x_valid - minimum) / rng

        # Neural network architecture parameters
        input_layer_size = n
        hidden_layer_1_size = 100
        hidden_layer_2_size = 100
        num_labels = 2
        lambda_ = 0

        # Gradient checking
        check_nn_gradients()

        # Set the models's architecture (the size of each layer)
        architecture = [input_layer_size, hidden_layer_1_size,
                        hidden_layer_2_size, num_labels]
        output_layer = 'softmax'
        mlp = MultiLayerPerceptron(architecture, lambda_=lambda_,
                                   output_layer=output_layer)

        # Train the model using gradient descent
        J_train_history, J_valid_history = mlp.fit(x_train, y_train, x_valid,
                                                   y_valid, 0.05, 15000)

        # Calculate model predictions on the training and validation sets
        predictions_train = mlp.predict(x_train)
        predictions_valid = mlp.predict(x_valid)

        # Calculate the mean squared error on the training and validation sets
        mse_train = mlp.mse(predictions_train, y_train)
        mse_valid = mlp.mse(predictions_valid, y_valid)
        print(f'\n> Training set loss (mean squared error) : {mse_train:.4f}')
        print(f'> Validation set loss (mean squared error) : {mse_valid:.4f}')

        # Calculate training set accuracy
        acc = np.mean(predictions_train == y_train)
        print(f'> Training set accuracy = {acc:.4f}')

        # Calculate validation set accuracy
        acc = np.mean(predictions_valid == y_valid)
        print(f'> Validation set accuracy = {acc:.4f}')

        # Save model topology and weights
        print("> saving model './saved_model.npy' to disk...")
        with open('saved_model.npy', 'wb') as f:
            np.save(f, minimum)
            np.save(f, rng)
            np.save(f, architecture)
            np.save(f, output_layer)
            np.save(f, lambda_)
            np.save(f, mlp.theta)

        # Plot the learning curves
        plt.plot(J_train_history, label='Training Loss')
        plt.plot(J_valid_history, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
