import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn, append_bias_term


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
        self._weight = None


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        biased_training_data = training_data
        biased_training_data = append_bias_term(biased_training_data)
        D = biased_training_data.shape[1]
        C = get_n_classes(training_labels)
        label_onehot = label_to_onehot(training_labels)
        self._weight = np.random.normal(0, 0.1, (D, C))
        for i in range(self.max_iters):
            self._weight -= self.lr * self.__gradient(biased_training_data, label_onehot, self._weight)
            pred_labels = self.__predicition(biased_training_data, self._weight)
            if accuracy_fn(pred_labels, training_labels) == 100:
                return pred_labels
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        biased_test_data = test_data
        biased_test_data = append_bias_term(biased_test_data)
        return self.__predicition(biased_test_data, self._weight)

    def __softmax(self, data, w):
        """
        Compute the softmax of the data.

        Arguments:
            data (array): of shape (N,D)
            w (array): of shape (D,C)
        Returns:
            (array): of shape (N,C)
        """

        return np.exp(data @ w) / np.sum(np.exp(data @ w), 1)[:, None]

    def __gradient(self, data, labels, w):
        """
        Compute the gradient of the loss.

        Arguments:
            data (array): of shape (N,D)
            labels (array): of shape (N,C)
            w (array): of shape (D,C)
        Returns:
            (array): of shape (D,C)
        """
        return data.T @ (self.__softmax(data, w) - labels)

    def __predicition(self, data, w):
        """
        Compute the predicted labels.

        Arguments:
            data (array): of shape (N,D)
            w (array): of shape (D,C)
        Returns:
            (array): of shape (N,)
        """
        return onehot_to_label(self.__softmax(data, w))




