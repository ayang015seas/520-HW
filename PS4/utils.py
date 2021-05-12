import numpy as np
import os
import matplotlib.pyplot as plt

"""
TLDR:

load_data_from_txt_file
    - Load data into a numpy array from a text file (You'll probably
    not have to use this directly)

classification_error
    - Compute error rate

mean_squared_error
    - Compute mean squared error

load_all_train_test_data
    - Load the train and test data from a specified FOLDER

load_learning_curve_data
    - Load the learning curve data (10%, 20%, ...) from a specified
    FOLDER

load_all_cross_validation_data
    - Load all the cross validation data in folds from a specified
    FOLDER

partition_cross_validation_fold
    - Separate cross validation folds into training and testing set.
    Often used with `load_all_cross_validation_data`

augment_bias
    - Add a bias dimension

load_initial_weights
    - Load the initial weights for neural network from a specified FOLDER

plot_reconstruction
    - Plot 10%, 20%, ..., 100% image reconstructions from PCA

"""



def load_data_from_txt_file(file_name, is_label=False):
    """
    Load data from a text file

    Args:
        file_name (str):
            directory to the text file

        is_label (bool):
            whether the text file is a label file 
            default is False

    Returns:
        type: np.matrix
        A data matrix of size (N, d) is the number of data points
        and d is the number of dimensions to represent each data point. 
        If is_label is True, then d is 0 and the returned data is a vector
        with size (N,)
    """
    data = []
    with open(file_name, "r") as f:
        for line in f:
            line_data = None
            if is_label:
                line_data = float(line.strip())
            else:
                if "," in line.strip():
                    line_data = [float(x) for x in line.strip().split(",")]
                else:
                    line_data = [float(x) for x in line.strip().split("\t")]

            data.append(line_data)
    data = np.array(data, dtype=np.float64)
    return data


def classification_error(y_pred, y_true):
    """
    Compute classification error between y_pred (prediction) and y_true (true label)

    Args: 
        y_pred (np.array)
        N dimensional vector of {+1, -1}

        y_true (np.array):
        N dimensional vector of {+1, -1}
    Returns: 
        error (float):
        The portion of y_true where y_pred != y_true (classification error rate)
    """
    assert(y_pred.shape == y_true.shape)
    error = float(np.sum(y_pred.flatten() != y_true.flatten()))/len(y_true)
    return error


def mean_squared_error(y_pred, y_true):
    """ Compute mean squared error
    Args: 
        y_pred (np.array)
            N dimensional vector

        y_true (np.array):
            N dimensional vector

    Returns: 
        error (float): mean squared error
    """
    assert(y_pred.shape == y_true.shape)
    return np.mean((y_pred-y_true)**2)


def load_all_train_test_data(folder):
    """
    Load all train and test data

    Args:
        folder (str):
            Directory to the folder containing the data
            This folder must contain 4 files
                X_test.txt
                y_train.txt
                X_test.txt
                y_test.txt

    Returns:
        (X_train, y_train, X_test, y_test)
            X_train is a np.array of size (N, d)
            X_test is a np.array of size (M, d)
            y_train is a np.array of size (N,)
            y_test is a np.array of size (M,)

    Example usage:
        >>> X_train, y_train, X_test, y_test = load_all_train_test_data("path/to/data/folder")

    """
    X_train_file = f"{folder}/X_train.txt"
    y_train_file = f"{folder}/y_train.txt"
    X_test_file  = f"{folder}/X_test.txt"
    y_test_file  = f"{folder}/y_test.txt"
    X_train = load_data_from_txt_file(X_train_file)
    y_train = load_data_from_txt_file(y_train_file, True)
    X_test = load_data_from_txt_file(X_test_file)
    y_test = load_data_from_txt_file(y_test_file, True)
    return (X_train, y_train, X_test, y_test)


def load_learning_curve_data(learning_curve_data_folder):
    """
    Load all learning curve data
    
    Args:
        learning_curve_data_folder (str):
            Directory to the folder containing the data. This
            folder must contain the following files
                X_train_10%.txt, X_train_20%.txt,..., X_train_100%.txt
                y_train_10%.txt, y_train_10%.txt,..., y_train_100%.txt

    Returns:
        (all_X_train, all_y_train)
            all_X_train is a list of 10 numpy arrays, with increasing
            number of rows but with the same number of columns (features)

            all_y_train is a list of 10 numpy arrays, with increasing number
            of elements

    Example usage
        >>> subsets_X, subsets_y = load_learning_curve_data("/path/to/folder/with/data")
        >>> for i, X in enumerate(subsets_X):
        >>>     y = subsets_y[i]
        >>>     # Train on X and y
    """    
    all_X_train = []
    all_y_train = []
    
    for percent in range(10, 101, 10): # For percent from 10, 20, ..., 100
        X_file = f"{learning_curve_data_folder}/X_train_{percent}%.txt"
        y_file = f"{learning_curve_data_folder}/y_train_{percent}%.txt"
        X = load_data_from_txt_file(X_file)
        y = load_data_from_txt_file(y_file, True)
        all_X_train.append(X)
        all_y_train.append(y)
    
    return (all_X_train, all_y_train)


def load_all_cross_validation_data(validation_data_folder):
    """
    Load all data to do cross validation experiment

    Args:
        validation_data_folder (str):
            Directory to the folder containing the data
            This directory must contains 5 sub-directories:
                Fold1
                Fold2
                Fold3
                Fold4
                Fold5

    Returns:
        all_folds (list)
            all_folds is a list of 5 elements. Each element is
            a tuple (X,y) where
            X is a numpy array of shape (N, d)
            y is a numpy array of shape (N,)

    Example usage:
        >>> all_folds = load_all_cross_validation_data("/path/to/folder/with/CV-data")
        >>> fold_number = 2 # Pick fold number 3 as leave out fold
        >>> leave_out_data, training_data = partition_cross_validation_fold(all_folds, fold_number)

    """
    all_folds = []

    for fold in [1,2,3,4,5]:
        X_file = f"{validation_data_folder}/Fold{fold}/X.txt"
        y_file = f"{validation_data_folder}/Fold{fold}/y.txt"
        X = load_data_from_txt_file(X_file)
        y = load_data_from_txt_file(y_file, True)
        all_folds.append((X,y))
    return all_folds


def partition_cross_validation_fold(all_folds, fold):
    """
    Treat the specified fold as leave out data and concatenate the 
    remaining folds into training data

    Args:
        all_folds (list):
            list of data folds. Assuming this is returned by
            `load_all_cross_validation_data`
        
        fold (int):
            The index of the fold to be treated as leave out data

    Returns:
        (leave_out_data, training_data)
            leave_out_data is a tuple of two elements (X_lo, y_lo)
                X_lo is a numpy array of size (N,d)
                y_lo is a numpy array of size (N,)
            
            training_data is a tuple of two elements (X_tr, y_tr)
                X_tr is a numpy array of size (M,d)
                y_tr is a numpy array of size (M,d)

                where M is approximately 4x N (assuming we're doing 5 fold
                cross validation)
    
    Example usage:
        >>> all_folds = load_all_cross_validation_data("/path/to/folder/with/CV-data")
        >>> fold_number = 2 # Pick fold number 3 as leave out fold
        >>> leave_out_data, training_data = partition_cross_validation_fold(all_folds, fold_number)

    """
    leave_out_data = all_folds[fold]
    training_data  = [all_folds[num] for num in range(len(all_folds)) if num != fold]
    
    # Concatenate all the remaining data
    all_x  = np.concatenate([portion[0] for portion in training_data])
    all_y  = np.concatenate([portion[1] for portion in training_data])
    
    training_data = (all_x, all_y)
    return leave_out_data, training_data


def augment_bias(X):
    """
    Add a "bias dimension" to the data by simply adding a column of 
    1 to the feature matrix

    Args:
        X (numpy.array)
            X has size (N,d)
    Returns:
        X_augment (numpy.array)
            X_augment has size (N, d+1)
    """
    return np.concatenate((X, np.ones((X.shape[0],1))), axis=1)

def load_initial_weights(folder):
    """
    Load the initial weights to initialize neural networks

    Arg:
        folder (str): Path to folder containing the initial
            weights value
    
    Returns:
        A dictionary whose (field_name, value) pairs correspond
            to parameter name and initial value

    Example usage:
        >>> init = utils.load_initial_weights("path/to/folder/with/initial/weights")
        >>> net  = NeuralNetworkClassification(input_dim, num_hidden=5,
        >>>         activation="sigmoid", **init)
    
    """
    b1_file = os.path.join(folder, "b1.txt")
    b2_file = os.path.join(folder, "b2.txt")
    W1_file = os.path.join(folder, "W1.txt")
    W2_file = os.path.join(folder, "W2.txt")
    
    b1 = load_data_from_txt_file(b1_file, True)
    b2 = load_data_from_txt_file(b2_file, True)[0] # A scalar
    W1 = load_data_from_txt_file(W1_file)
    W2 = load_data_from_txt_file(W2_file)
        
    return {"b1":b1, "b2":b2, "W1":W1, "W2":W2}

def plot_reconstruction(images, title, filename):
    '''
    Plots 10%, 20%, ..., 100% reconstructions of a 28x28 image

    Args
        images (numpy.array)
            images has size (10, 28, 28)
        title (str)
            title within the image
        filename (str)
            name of the file where the image is saved

    Returns
        None

    Example usage:
        >>> images = np.zeros(10,28,28)
        >>> images[0,:,:] = x10.reshape((28,28))
        >>> images[1,:,:] = x20.reshape((28,28))
        >>> ...
        >>> images[9,:,:] = x100.reshape((28,28))
        >>> utils.plot_reconstruction(images, 'Image Title', 'filename.png')
    '''
    assert images.shape == (10,28,28)
    fig, (
        (ax0, ax1, ax2, ax3),
        (ax4, ax5, ax6, ax7),
        (ax8, ax9, _, _)
    ) = plt.subplots(3, 4)
    axes = [ax9, ax8, ax7, ax6, ax5, ax4, ax3, ax2, ax1, ax0]
    percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(10):
        ax = axes[i]
        percent_name = f'{percents[i]}%' if i != 9 else 'Original'
        ax.set(title=percent_name)
        axes[i].imshow(images[i,:,:], cmap='gray')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
