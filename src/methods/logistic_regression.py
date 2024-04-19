import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, task_kind = "classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind
        self.weight = None


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        D = training_data.shape[1]
        C = get_n_classes(training_labels)
        label_onehot = label_to_onehot(training_labels)
        self.weight = np.zeros([D, C])
        for _ in range(self.max_iters):
            self.weight -= self.lr * self.__gradient__(training_data, label_onehot, self.weight)
            pred_labels = self.__predicition__(training_data, self.weight)
            if accuracy_fn(pred_labels, training_labels) == 100:
                break
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return self.__predicition__(test_data, self.weight)

    def __softmax__(self, data, w):
        """
        Compute the softmax of the data.

        Arguments:
            data (array): of shape (N,D)
            w (array): of shape (D,C)
        Returns:
            (array): of shape (N,C)
        """
        return np.exp(data @ w) / np.sum(np.exp(data @ w), 1)[:, None]

    def __gradient__(self, data, labels, w):
        """
        Compute the gradient of the loss.

        Arguments:
            data (array): of shape (N,D)
            labels (array): of shape (N,C)
            w (array): of shape (D,C)
        Returns:
            (array): of shape (D,C)
        """
        return data.T @ (self.__softmax__(data, w) - labels)

    def __predicition__(self, data, w):
        """
        Compute the predicted labels.

        Arguments:
            data (array): of shape (N,D)
            w (array): of shape (D,C)
        Returns:
            (array): of shape (N,)
        """
        return np.argmax(self.__softmax__(data, w), 1)




