# Imports.
import numpy as np


#########################################################
# Set the hyperparameters for the Neural Network to use.
#########################################################
iterations = int(3e3)
learning_rate = 5e-1
hidden_nodes = 25
output_nodes = 1


class NeuralNetwork(object):
    """ Class implementation of an Artificial Neural Network (ANN).
    """
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """ Initialize an ANN instance.
        
        Params
        ======
            input_nodes (int): Number of nodes in the input layer
            hidden_nodes (int): Number of nodes in the hidden layer
            output_nodes (int): Number of nodes in the output layer
            learning_rate (float): Learning rate
        """
        
        # Set the number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights.
        self.weights_input_to_hidden = np.random.normal(
            0.,
            self.input_nodes**-.5, 
            (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(
            0.,
            self.hidden_nodes**-.5,
            (self.hidden_nodes, self.output_nodes))
        
        # Set the learning rate.
        self.lr = learning_rate
        
        # Setting the activation function to a sigmoid function.
        # Here, we define a function with a lambda expression.
        self.activation_function = lambda x : 1. / (1. + (np.exp(-x)))
                            

    def train(self, features, targets):
        """ Train the ANN on batch of features and targets. 
        
        Params
        ======
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        """
        
        # Set the parameters.
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        # The (i) is for input, and (o) is for output.
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
            
        # Update the weights.
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        """ Perform forward passing.
         
        Params
        ======
            X: Features batch

        """
        
        # Forward passing calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        # Return both final_output, and hidden_output layers.
        return final_outputs, hidden_outputs

    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        """ Perform back-propagation.
         
        Params
        ======
            final_outputs: Output from forward pass
            y: Target (i.e. label) batch
            delta_weights_i_h: Change in weights from input to hidden layers
            delta_weights_h_o: Change in weights from hidden to output layers

        """
        
        # Backward passing calculations.
        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs
        
        # Hidden layer's contribution to the error.
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        
        # Backpropagated error terms.
        output_error_term = error
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden).
        delta_weights_i_h += hidden_error_term * X[:, None]
        
        # Weight step (hidden to output).
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        
        # Return the delta weights.
        return delta_weights_i_h, delta_weights_h_o

    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update weights on gradient descent step.
         
        Params
        ======
            delta_weights_i_h: Change in weights from input to hidden layers
            delta_weights_h_o: Change in weights from hidden to output layers
            n_records: Number of records

        """
        
        # Update hidden-to-output weights with gradient descent step.
        self.weights_hidden_to_output += (self.lr * delta_weights_h_o) / n_records
        
        # Update input-to-hidden weights with gradient descent step.
        self.weights_input_to_hidden += (self.lr * delta_weights_i_h) / n_records
        
        
    def run(self, features):
        """ Run a forward pass through the network with input features.
        
        Params
        ======
            features: 1D array of feature values
        """
        
        ## Forward passing calculations.
        #
        # Hidden layer calculations.
        #
        # Signals into hidden layer.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        
        # Signals from hidden layer.
        hidden_outputs = self.activation_function(hidden_inputs)
        
        ## Output layer calculations.
        #
        # Signals into final output layer.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        
        # Signals from final output layer.
        final_outputs = final_inputs
        
        # Return the final outputs
        return final_outputs
