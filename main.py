import argparse

import numpy as np
import time
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')

        # (2938, 5), (327, 5), (2938,), (327,), (2938, 2), (327, 2)
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)



    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Arbitrary value for the fraction of the training data used for validation
        fraction_validation_test = 0.3

        # Split the training data into training and validation for regression task (center_locating)
        c_num_samples = ctrain.shape[0]
        rinds = np.random.permutation(c_num_samples)
        n_validation = int(c_num_samples * fraction_validation_test)
        ctest = ctrain[rinds[:n_validation]] 
        ctrain = ctrain[rinds[n_validation:]]

        # Split the training data into training and validation for classification task (breed_identifying)
        xy_num_samples = xtrain.shape[0]
        rinds = np.random.permutation(xy_num_samples)
        n_validation = int(xy_num_samples * fraction_validation_test)
        xtest = xtrain[rinds[:n_validation]]
        ytest = ytrain[rinds[:n_validation]] 
        xtrain = xtrain[rinds[n_validation:]]
        ytrain = ytrain[rinds[n_validation:]]
        pass

    # Normalize the data, can be disabled with the argument --no_norm
    if(args.no_norm == False):
        x_means = xtrain.mean(0,keepdims=True)
        x_stds = xtrain.std(0,keepdims=True)
        xtrain = normalize_fn(xtrain, x_means, x_stds)
        xtest = normalize_fn(xtest, x_means, x_stds)
    
    ### WRITE YOUR CODE HERE to do any other data processing

    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    results = np.zeros((0,2))
    possible_k = np.arange(1, 50 if args.test_hyperparam else 2)
    possible_lmda = np.arange(0,700,10)
    

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "linear_regression":
        possible_hyp = possible_lmda
        if(args.test_hyperparam):
            for lmda in possible_lmda:
                print(f"\n------------- Lambda = {lmda} -------------")
                method_obj = LinearRegression(lmda=lmda, task_kind=args.task)
                results = np.append(results, trainAndEvaluate(method_obj, xtrain, xtest, ytrain, ytest, ctrain, ctest), axis=0)
        else :
            method_obj = LinearRegression(lmda=args.lmda, task_kind=args.task)
        
    elif args.method == "logistic_regression":  ### WRITE YOUR CODE HERE
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters, task_kind=args.task)
    elif args.method == "knn":
        possible_hyp = possible_k
        if(args.test_hyperparam):
            for k in possible_k:
                print(f"\n------------- K = {k} -------------")
                method_obj = KNN(k=k, task_kind=args.task)
                results = np.append(results, trainAndEvaluate(method_obj, xtrain, xtest, ytrain, ytest, ctrain, ctest), axis=0)
        else :
            method_obj = KNN(k=args.K, task_kind=args.task)
    else:
        raise Exception("Invalid choice of method! Please choose one of the following: dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
        
    

    ## 4. Train and evaluate the method
    if(args.test_hyperparam):
        regression = args.task == "center_locating"
        plt.plot(possible_hyp, results[:,0], label='Training')
        plt.plot(possible_hyp, results[:,1], label='Test')

        # Adding title
        plt.title('Result of ' +('Center Locating' if regression else 'Breed Identifying') + ' with ' + args.method + ' method')

        # Adding labels
        plt.xlabel('K')
        plt.ylabel('Mean Square Error' if regression else 'Accuracy [%]')

        plt.legend()
        plt.grid()
        plt.show()  
    else:
        trainAndEvaluate(method_obj, xtrain, xtest, ytrain, ytest, ctrain, ctest)

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    if (args.hyperpar_logistic):
        it = np.linspace(50, 600, 12)
        results = np.zeros(0)
        array = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        for lr in array:
            print(f"\n------------- lr = {lr} -------------")
            for i in range(50, 650, 50):
                print(f"\n------------- max_iters = {i} -------------")
                method_obj = LogisticRegression(lr=lr, max_iters=i, task_kind=args.task)
                results = np.append(results, np.mean(trainAndEvaluate(method_obj, xtrain, xtest, ytrain, ytest, ctrain, ctest)))

            plt.plot(it, results, label='Learning Rate = ' + str(lr))
            results = np.zeros(0)

        # Adding title
        plt.title('The best hyperparameters with Logistic Regression')

        # Adding labels
        plt.xlabel('Number of iterations')
        plt.ylabel('Mean Square Error' if args.task == "center_locating" else 'Accuracy [%]')

        plt.legend()
        plt.show()

def trainAndEvaluate(method_obj, xtrain, xtest, ytrain, ytest, ctrain, ctest):

    if args.task == "center_locating":
        # Fit parameters on training data
        preds_train = method_obj.fit(xtrain, ctrain)

        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.5f} - Test loss = {loss:.5f}")

        return np.array([train_loss, loss]).reshape(1,2)

    elif args.task == "breed_identifying":
        s1 = time.time()
        # Fit (:=train) the method on the training data for classification task
        preds_train = method_obj.fit(xtrain, ytrain)
        s2 = time.time()
        if(args.time):
            print(f"Training time: {s2 - s1:.2f}s for{args.method}")
        # Predict on unseen data
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc_train = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1:.6f}")

        acc_test = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc_test:.3f}% - F1-score = {macrof1:.3f}")

        return np.array([acc_train, acc_test]).reshape(1,2)
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!
    parser.add_argument('--no_norm', action="store_true", help="disable data normalization")
    parser.add_argument('--test_hyperparam', action="store_true", help="vary hyperparameters and plot a graph of the results")

    parser.add_argument('--hyperpar_logistic', type=bool, default=False, help="Boolean")
    parser.add_argument('--time', type=bool, default=False, help="Show time")

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
