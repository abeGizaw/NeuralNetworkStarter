import time
from abc import abstractmethod

import numpy as np
import nnfs
nnfs.init()

class Model:
    def __init__(self):
        self.layers = []
        self.input_layer = None
        self.trainable_layers = []

        self.accuracy = None
        self.optimizer = None
        self.loss = None
        self.softmax_classifier_output = None
        self.output_layer_activation = None

    def add(self, layer):
        self.layers.append(layer)

    # Asterisk requires loss and optimizer to be specified when set is called
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy


    # Finalizes the model, setting the next and previous layers
    def finalize(self):
        """
        Connects all the layers, setting their next and previous
        Keeps track of all trainable layers
        :return:
        """
        self.input_layer = InputLayer()
        layer_count = len(self.layers)

        for i in range(layer_count):
            # If it is the 1st layer then
            # the prev layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # Hidden layer to hidden layer
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # If we are at the last layer, the next object is the loss func
            else:
                self.layers[i].prev = self.layers[i -1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If the layer has weights, then it is trainable
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # For regression models (not shown atm)
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If the model uses softmax and categorical cross entropy
        # Save off the combined model for faster backward pass
        if isinstance(self.layers[-1], ActivationSoftmax) and \
            isinstance(self.loss, LossCategoricalCrossEntropy):
            self.softmax_classifier_output = \
                ActivationSoftmax_Loss_CategoricalCrossEntropy()
   

    def train(self, X, y, *, epochs = 1, print_every = 1, batch_size = None):
        """
        :param X: train data inputs (# of inputs x flattened input)
        :param y: the trained data expected
        :param epochs: iterations
        :param batch_size
        :param print_every
        :return:
        """
        print(f'starting to train\n')

        # Initialize accuracy object
        self.accuracy.init(y)

        # Calculate number of steps
        train_steps = 1
        if batch_size is not None:
            # Number of batches being trained
            train_steps = len(X) // batch_size

            # Add a step for leftover data
            if train_steps * batch_size < len(X):
                train_steps += 1
            print(f'running {len(X)} pieces of data in {train_steps} steps\n')


        for epoch in range(1, epochs + 1):
            epoch_time_s = time.time()
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                # Figure out data to train on
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                train_time_start = time.time()

                # Perform the forward pass
                output = self.forward(batch_X)

                # Calculate loss
                loss = self.loss.calculate(output, batch_y)

                # Get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Backward pass
                self.backward(output, batch_y)

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)

                total_train_time = time.time() - train_time_start
                if batch_size is not None and print_every is not None and step % print_every == 0 and step != 0:
                    print(f'step:{step}, acc:{accuracy:.3f}, loss:{loss:.3f}')
                    print(f'step took {total_train_time:.2f} seconds')

            total_epoch_time = time.time() - epoch_time_s
            epoch_data_loss = self.loss.calculate_accumulated()
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'epoch:{epoch}, acc:{epoch_accuracy:.3f}, loss:{epoch_data_loss:.3f}')
            print(f'epoch took {total_epoch_time:.2f} seconds\n')


    def validate(self, *, validation_data, batch_size = None, print_every = 100):
        """
        :param print_every:
        :param validation_data: inputs and expected out values
        :param batch_size
        :return:
        """
        print(f'starting to validate\n')

        self.loss.new_pass()
        self.accuracy.new_pass()
        X_val, y_val = validation_data

        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        for step in range(validation_steps):
            # Figure out data to train on
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            valid_time_s = time.time()
            output = self.forward(batch_X)
            loss = self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            valid_time_e = time.time()

            if batch_size is not None and print_every is not None and step % print_every == 0 and step != 0:
                print(
                    f'validation, step:{step}, acc:{accuracy:.3f}, loss:{loss:.3f}'
                )
                print(f'step took {valid_time_e - valid_time_s:.2f} seconds\n')

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f'final validation - acc:{validation_accuracy}, loss:{validation_loss}')


    def forward(self, X):
        # Pass in the input to the input layer
        self.input_layer.forward(X)

        # Call forward on every hidden layer
        # Output of prev layer is input of the next
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # The final layers output
        return self.layers[-1].output


    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # Sets the first dinputs
            self.softmax_classifier_output.backward(output, y)

            # Since we combined softmax and CrossEntropy, they will share dinputs
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward on every layer but the last in reversed order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
        # Is not a softmax classifier
        else:
            # This will set our initial dinputs
            self.loss.backward(output, y)

            # Go through layers in reverse order (backpropagation)
            # dinputs is now the parameter
            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)


            
class InputLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = inputs

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.inputs = None
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.dinputs = None
        self.dbiases = None
        self.dweights = None
 
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def __init__(self):
        self.inputs = None
        self.output = None

        self.dinputs = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    @staticmethod
    def predictions(outputs):
        return outputs


# Will Fill In Later
class ActivationSigmoid:
    @staticmethod
    def predictions(outputs):
        return (outputs > 0.5) * 1


class ActivationLinear:
    @staticmethod
    def predictions(outputs):
        return outputs


class ActivationSoftmax:
    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, inputs):
        # un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize for each sample
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array. Note dvalues is 2D
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Softmax jacobian array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient and add it to the array of sample gradients.
            # single_dvalues is a vector
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    @staticmethod
    def predictions(outputs):
        return np.argmax(outputs, axis=1)

class Loss:
    # Used for regularization loss. Not shown atm
    def __init__(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0
        self.trainable_layers = None

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, expectedOutput):
        sample_losses = self.forward(output, expectedOutput)
        data_loss = np.mean(sample_losses)

        # For batch/epoch statistics
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        return data_loss


    # Calculate accumulated loss
    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss


    # Reset variables for each new epoch
    def new_pass(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0


    @abstractmethod
    def forward(self, output, expectedOutput):
        pass


class LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.dinputs = None

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip data to prevent division by 0 and
        # dragging the mean towards a certain value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Sparse (Categorical) 
        if len(y_true.shape) == 1:
            # [[0,1,2], [y_true]]
            correct_conf = y_pred_clipped[range(samples), y_true]
        # One-hot
        elif len(y_true.shape) == 2:
            correct_conf = np.sum(y_pred_clipped*y_true, axis = 1)
        else:
            raise ValueError("Wrong dimensions for y_true")

        neg_log_likelihood = -np.log(correct_conf)
        return neg_log_likelihood


    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # Number of labels in every sample
        labels = len(dvalues[0])

        # if labels are spare, turn them into a one hot vector
        if len(y_true.shape) == 1:
            # eye creates identity matrix
            y_true = np.eye(labels)[y_true]

        # Calculate Gradient
        self.dinputs = -y_true/dvalues

        # Normalize Gradient
        self.dinputs = self.dinputs / samples

# Combined softmax and cross entropy for faster backward step
class ActivationSoftmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.dinputs = None

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # Calculate Gradients (Taking advantage of one-hot encoded y-true)
        self.dinputs[range(samples), y_true] -= 1

        # Normalize
        self.dinputs /= samples

class OptimizerSGD:
    def __init__(self, learning_rate = 1.0):
        self.learning_rate = learning_rate

    # Given layer of an object, this will adjust the weights and biases 
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Accuracy:

    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    # Calculate accumulated accuracy
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    # Reset variables for each new epoch
    def new_pass(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0


    @abstractmethod
    def compare(self, predictions, y):
        pass


class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary=False):
        super().__init__()
        self.binary = binary

    # Return a list of true and false values (1s and 0s)
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y