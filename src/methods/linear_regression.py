import numpy as np
import sys
from ..utils import append_bias_term

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = "regression"

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        # Add a column of ones to the training data to account for the bias term
        X = append_bias_term(self.training_data)
        # Calculate the weights
        I = np.eye(X.shape[1])
        # Set the first element of the identity matrix to 0 because we don't want to regularize the bias term
        I[0,0] = 0
        # Calculate the weights
        self.w = np.linalg.inv(X.T@X + self.lmda*I)@X.T @self.training_labels


    def predict(self, test_data):
            """
                Runs prediction on the test data.
                
                Arguments:
                    test_data (np.array): test data of shape (N,D)
                Returns:
                    test_labels (np.array): labels of shape (N,regression_target_size)
            """

            # Add a column of ones to the test data
            X = append_bias_term(test_data)
            # Calculate the predicted regression targets
            pred_regression_targets = X@self.w


            return pred_regression_targets
