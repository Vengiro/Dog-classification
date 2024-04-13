import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.predict(training_data)

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """

        if self.task_kind == "classification":
            test_labels = np.array([self.__knn_one_classification(one) for one in test_data])
        else:
            test_labels = np.array([self.__knn_one_regression(one) for one in test_data])

        
        return test_labels
    
    def __knn_one_classification(self, one):
        """Predict the label of a single example using the k-nearest neighbors algorithm.

        Inputs:
            one: shape (D,)
        Outputs:
            best_label_id: integer
        """

        # Get the indices of the k nearest neighbors
        nn_indices = self.__find_k_nearest_neighbors(one)

        # Get the labels of the k nearest neighbors
        neighbor_labels = self.training_labels[nn_indices]

        # Return the most common label
        return np.argmax(np.bincount(neighbor_labels))
    
    def __knn_one_regression(self, one):
        """Predict the label of a single example using the k-nearest neighbors algorithm.

        Inputs:
            one: shape (D,)
        Outputs:
            best_label_id: integer
        """

        # Get the indices of the k nearest neighbors
        nn_indices = self.__find_k_nearest_neighbors(one)

        # Get the labels of the k nearest neighbors
        neighbor_coords = self.training_labels[nn_indices]

        # Return the average coords of the k nearest neighbors
        return np.mean(neighbor_coords, axis=0, keepdims=True)

    
    def __euclidean_dist(self, one):
        """Compute the Euclidean distance between a single one
        vector and all vectors in a the training_data.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        return np.sqrt(((self.training_data - one) ** 2).sum(axis=1))
    
    def __find_k_nearest_neighbors(self, one):
        """ Find the indices of the k nearest neighbors by computing distances.

        Inputs:
            k: integer
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """

        # Compute distances
        distances = self.__euclidean_dist(one) 

        indices = np.argsort(distances)[:self.k]
        return indices