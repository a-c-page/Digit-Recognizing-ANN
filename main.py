import numpy as np
import matplotlib.pyplot as plt


def print_data_set(data_set):
    for num in data_set:
        for x in range(0, 45, 5):
            for y in range(5):
                if num[x + y] == 1:
                    print(' # ', end="")
                else:
                    print('   ', end="")
            print('')
        print('\n')


# Initializes weight/bias matrices with random digits from in range [-0.5, 0.5]
def init_params():
    weights_1 = np.random.rand(5, 45) - 0.5
    biases_1 = np.random.rand(5, 1) - 0.5
    weights_2 = np.random.rand(10, 5) - 0.5
    biases_2 = np.random.rand(10, 1) - 0.5
    return weights_1, biases_1, weights_2, biases_2


# Softmax function - used to turn a set into probabilities
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(weights_1, biases_1, weights_2, biases_2, X):
    combined_layer_1 = weights_1.dot(
        X) + biases_1  # [5 x 45] DOT [45 x 10] + [5 x 1] = [5 x 10] where each column is an example
    activated_layer_1 = ReLU(combined_layer_1)  # Activation on layer 1
    combined_layer_2 = weights_2.dot(
        activated_layer_1) + biases_2  # [10 x 5] DOT [5 X 10] + [10 x 1] = [10 x 10] where each column is an example
    activated_layer_2 = softmax(combined_layer_2)  # Softmax to get probability set
    return combined_layer_1, activated_layer_1, combined_layer_2, activated_layer_2


# ReLU (Rectified linear unit) - used as the activation function
def ReLU(Z):
    return np.maximum(Z, 0)


# Derivative of ReLU - used during back propagation
def dReLU(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(combined_layer_1, activated_layer_1, combined_layer_2, activated_layer_2, weights_1, weights_2, X, Y):
    one_hot_Y = one_hot(Y)
    dcombined_layer_2 = activated_layer_2 - one_hot_Y
    dweights_2 = 1 / columns * dcombined_layer_2.dot(activated_layer_1.T)
    dbiases_2 = 1 / columns * np.sum(dcombined_layer_2)
    dcombined_layer_1 = weights_2.T.dot(dcombined_layer_2) * dReLU(combined_layer_1)
    dweights_1 = 1 / columns * dcombined_layer_1.dot(X.T)
    dbiases_1 = 1 / columns * np.sum(dcombined_layer_1)
    sum_square = np.sum(np.square(activated_layer_2 - one_hot_Y))
    return dweights_1, dbiases_1, dweights_2, dbiases_2, sum_square


def update_params(weights_1, biases_1, weights_2, biases_2, dweights_1, dbiases_1, dweights_2, dbiases_2, alpha):
    weights_1 = weights_1 - alpha * dweights_1
    biases_1 = biases_1 - alpha * dbiases_1
    weights_2 = weights_2 - alpha * dweights_2
    biases_2 = biases_2 - alpha * dbiases_2
    return weights_1, biases_1, weights_2, biases_2


def get_predictions(activated_layer_2):
    return np.argmax(activated_layer_2, 0)


def gradient_descent(X, Y, alpha, epochs):
    weights_1, biases_1, weights_2, biases_2 = init_params()
    sse_array = []
    for i in range(epochs):
        combined_layer_1, activated_layer_1, combined_layer_2, activated_layer_2 = forward_prop(weights_1, biases_1,
                                                                                                weights_2, biases_2, X)
        dweights_1, dbiases_1, dweights_2, dbiases_2, sum_square = backward_prop(combined_layer_1, activated_layer_1,
                                                                                 combined_layer_2, activated_layer_2,
                                                                                 weights_1,
                                                                                 weights_2, X, Y)
        weights_1, biases_1, weights_2, biases_2 = update_params(weights_1, biases_1, weights_2, biases_2, dweights_1,
                                                                 dbiases_1, dweights_2, dbiases_2, alpha)
        sse_array.append(sum_square)
    return weights_1, biases_1, weights_2, biases_2, sse_array


def make_predictions(X, weights_1, biases_1, weights_2, biases_2):
    _, _, _, activated_layer_2 = forward_prop(weights_1, biases_1, weights_2, biases_2, X)
    predictions = get_predictions(activated_layer_2)
    return predictions


def test_prediction(index, weights_1, biases_1, weights_2, biases_2):
    current_set = testing_data_values[:, index, None]
    prediction = make_predictions(testing_data_values[:, index, None], weights_1, biases_1, weights_2, biases_2)
    label = testing_data_labels[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    if prediction == label:
        correctness = 1
    else:
        correctness = 0
    print_data_set([current_set])
    return correctness


# MAIN FUNCTION
if __name__ == '__main__':
    # np.set_printoptions(formatter={'float_kind': '{:f}'.format})  # Used to change numpy print format

    # DATA SETS
    training_data = [
        [0,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # ZERO
        [1,
         0, 0, 1, 0, 0,
         0, 1, 1, 0, 0,
         1, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0],  # ONE
        [2,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 1, 0, 0,
         0, 1, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1],  # TWO
        [3,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # THREE
        [4,
         0, 0, 0, 1, 0,
         0, 0, 1, 1, 0,
         0, 0, 1, 1, 0,
         0, 1, 0, 1, 0,
         0, 1, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 1, 1, 1, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0],  # FOUR
        [5,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # FIVE
        [6,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # SIX
        [7,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 1, 0, 0, 0,
         0, 1, 0, 0, 0,
         0, 1, 0, 0, 0],  # SEVEN
        [8,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # EIGHT
        [9,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0]  # NINE
    ]  # TRAINING DATA (10 examples, 0-9)
    testing_data = [
        [0,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 1, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 1, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 1, 1,
         0, 1, 1, 1, 0],  # ZERO - 1
        [0,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 1, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 1, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 1],  # ZERO - 2
        [0,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1],  # ZERO - 3
        [1,
         1, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         1, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         1, 1, 1, 1, 1],  # ONE - 1
        [1,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0],  # ONE - 2
        [1,
         0, 0, 1, 0, 0,
         0, 1, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 1, 1, 1, 0],  # ONE - 3
        [2,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 1, 0, 0,
         0, 1, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1],  # TWO - 1
        [2,
         1, 1, 1, 1, 0,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 1, 0, 0,
         0, 1, 0, 0, 1,
         1, 0, 0, 0, 0,
         1, 1, 1, 0, 1],  # TWO - 2
        [2,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         1, 0, 0, 0, 0,
         0, 1, 0, 1, 0,
         1, 0, 0, 0, 0,
         0, 1, 1, 1, 1],  # TWO - 3
        [3,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 1, 1, 1, 1],  # THREE - 1
        [3,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # THREE - 2
        [3,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 1, 0, 1,
         0, 0, 1, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # THREE - 3
        [4,
         0, 0, 0, 1, 0,  # FOUR
         0, 0, 1, 1, 0,
         0, 0, 1, 1, 0,
         0, 1, 0, 1, 0,
         0, 1, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 1, 1, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 0, 0],  # FOUR - 1
        [4,
         0, 0, 0, 1, 0,  # FOUR - 1
         0, 0, 0, 1, 0,
         0, 0, 1, 1, 0,
         0, 1, 0, 1, 0,
         0, 1, 0, 1, 0,
         1, 0, 1, 1, 0,
         1, 1, 0, 1, 0,
         0, 0, 0, 1, 0,
         1, 0, 0, 1, 0],  # FOUR - 2
        [4,
         1, 0, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 0, 0, 1, 0,
         1, 1, 1, 1, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0],  # FOUR - 3
        [5,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1],  # FIVE - 1
        [5,
         1, 1, 1, 1, 0,  # FIVE - 1
         1, 0, 0, 0, 0,
         1, 0, 1, 0, 0,
         1, 1, 0, 1, 0,
         1, 0, 0, 0, 1,
         0, 0, 1, 0, 1,
         0, 1, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # FIVE - 2
        [5,
         1, 1, 1, 1, 1,  # FIVE
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # FIVE - 3
        [6,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # SIX - 1
        [6,
         0, 1, 0, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 1, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 1, 1, 0],  # SIX - 2
        [6,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0],  # SIX - 3
        [7,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0],  # SEVEN - 1
        [7,
         0, 1, 1, 1, 1,
         0, 0, 0, 0, 0,
         0, 0, 0, 1, 0,
         0, 0, 0, 1, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 1, 0, 0, 0,
         0, 1, 0, 1, 0,
         0, 0, 0, 0, 0],  # SEVEN - 2
        [7,
         1, 1, 1, 1, 1,
         0, 0, 0, 1, 1,
         0, 0, 1, 1, 0,
         0, 0, 1, 1, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 1, 1, 0, 0,
         0, 1, 1, 0, 0,
         0, 1, 1, 0, 0],  # SEVEN - 3
        [8,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1],  # EIGHT - 1
        [8,
         0, 1, 0, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 1, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 0, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 1, 0, 1,
         1, 0, 0, 0, 1,
         0, 1, 0, 1, 0],  # EIGHT - 2
        [8,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 0, 1, 0, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         0, 0, 1, 0, 0],  # EIGHT - 3
        [9,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         1, 1, 1, 1, 1],  # NINE - 1
        [9,
         0, 1, 1, 1, 0,
         0, 1, 0, 0, 1,
         0, 1, 0, 0, 1,
         0, 1, 0, 0, 1,
         0, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 1, 1, 1],  # NINE - 2
        [9,
         0, 1, 1, 1, 0,
         1, 0, 1, 0, 1,
         1, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 1, 0, 1,
         0, 0, 0, 0, 1,
         0, 1, 1, 1, 0]  # NINE - 3
    ]  # TESTING DATA (30 examples, 0-9 x 3)

    # TRAINING DATA OPERATIONS
    training_data_matrix = np.array(training_data)  # Converting into a numpy array [10 x 45]
    rows, columns = training_data_matrix.shape  # Obtaining the amount of rows and columns of training set [10 x 45]
    training_data_matrix = training_data_matrix[:].T  # Transposing matrix [45 x 10]
    training_data_labels = training_data_matrix[0]  # Labels in first column
    training_data_values = training_data_matrix[1:]  # Values from 2nd column onwards

    # TESTING DATA OPERATIONS
    testing_data_matrix = np.array(testing_data)  # Converting into a numpy array [30 x 45]
    testing_data_matrix = testing_data_matrix[:].T  # Transposing matrix [45 x 30]
    testing_data_labels = testing_data_matrix[0]  # Labels in first column
    testing_data_values = testing_data_matrix[1:]  # Values from 2nd column onwards

    weights_1, biases_1, weights_2, biases_2, sse_array = gradient_descent(training_data_values, training_data_labels,
                                                                           0.5, 500)

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.arange(500)
    y = sse_array

    plt.title("Sum-Squared-Error vs Epoch")
    plt.plot(x, y, color="red")

    answer1 = input("Would you like to see Sum-Squared-Error vs Epoch plotted? (y/n): ")

    if answer1.lower() == "y":
        plt.show()

    answer2 = input("Would you like to run these weights/biases on the testing set? (y/n): ")

    correct = 0
    if answer2.lower() == "y":
        for i in range(len(testing_data)):
            correct += test_prediction(i, weights_1, biases_1, weights_2, biases_2)
        print("Correctness % on test set: ", correct / 30 * 100, "%")
