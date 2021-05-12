# # Gaussian Mixture Models


import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multivariate_normal
from gmm import GaussianMixtureModel

X_test_all = utils.load_data_from_txt_file("P3/X_test.txt")
test_data = utils.load_data_from_txt_file("P3/X_test.txt")

def plot_contour_gaussian(ax, mean, covariance, eps=1e-2):
    """ Plot the contour of a 2d Gaussian distribution with given mean and 
    covariance matrix

    Args:
        ax (matplotlib.axes.Axes):
            Subplot used to plot the contour
        mean (numpy.array):
            Mean of the gaussian distribution
        covariance (numpy.array):
            Covariance matrix of the distribution
        eps:
            The cut off to draw the contour plot. The higher the value, 
            the smaller the contour plot.

    Returns:
        None

    """
    x1_range=np.linspace(-6, 8, 100)
    x2_range=np.linspace(-10, 8, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range, indexing='ij')
    Z      = np.concatenate((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis]), axis=1)
    P      = multivariate_normal.pdf(Z, mean, covariance)
    P[P < eps] = 0
    P      = P.reshape((len(x1_range), len(x2_range)))
    ax.contour(x1_range, x2_range, P.T, colors='black', alpha=0.2)
    

def plot_gmm_model(ax, learned_model, test_data, percent):
    """ Plot the learned GMM and its associated gaussian distribution
    against the test data

    Args:
        ax (matplotlib.axes.Axes):
            Subplot used to plot the contour

        learned_model (GaussianMixtureModel):
            A trained GMM
        
        test_data (numpy.float):
            The testing data

        percent (float):
            The percentage of training data, used to label the subplot

    Returns:
        None

    """
    for k in range(learned_model.K):
        plot_contour_gaussian(ax, learned_model.mus[k, :], learned_model.covariances[k, :, :])
    ax.scatter(test_data[:, 0], test_data[:, 1], alpha=0.5)
    ax.scatter(learned_model.mus[:, 0], learned_model.mus[:, 1], c="r")
    ax.set_ylim(-10, 8)
    ax.set_xlim(-6, 8)
    ax.set_title(f"{percent}%")


def plot_multiple_contour_plots(learned_models):
    """ Plot multiple learned GMMs

    Arg:
        learned_models (list):
            A list of learned models that were trained on 10%,
            20%, 30%, ..., 100% of training data
    
    Returns:
        fig:
            The figure handle which you can use to save the figure

    Example usage:
        >>> learned_models = ... # A list of trained GMMs trained on increasing data
        >>> fig = plot_multiple_contour_plots(learned_models)
        >>> fig.savefig("4(a)(ii).png)

    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))

    axes = axes.flatten()
    percentage_data = np.arange(10, 101, 10)
    for i, learned_model in enumerate(learned_models):
        plot_gmm_model(axes[i], learned_model, X_test_all, percentage_data[i])

    axes[-1].axis('off')
    axes[-2].axis('off')
    return fig


## Your code starts here

def load_means(learning_curve_data_folder):
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
    
    for percent in range(10, 101, 10): # For percent from 10, 20, ..., 100
        X_file = f"{learning_curve_data_folder}/mu_{percent}%.txt"
        X = utils.load_data_from_txt_file(X_file)
        all_X_train.append(np.asarray(X))    
    return np.asarray(all_X_train)

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
    
    for percent in range(10, 101, 10): # For percent from 10, 20, ..., 100
        X_file = f"{learning_curve_data_folder}/X_train_{percent}%.txt"
        X = utils.load_data_from_txt_file(X_file)
        all_X_train.append(np.asarray(X))    
    return np.asarray(all_X_train)


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
        X_file = f"{validation_data_folder}/X_train_fold{fold}.txt"
        y_file = f"{validation_data_folder}/X_test_fold{fold}.txt"
        X = utils.load_data_from_txt_file(X_file)
        y = utils.load_data_from_txt_file(y_file)
        all_folds.append((X,y))
    return all_folds

example_params = utils.load_data_from_txt_file("P3/MeanInitialization/Part_a/mu_10%.txt")
print(example_params)
# cov = []
# cov.append(np.identity(2))
# cov.append(np.identity(2))
# cov.append(np.identity(2))

# cov = np.asarray(cov)

init_means = load_means("P3/MeanInitialization/Part_a")
training_data = load_learning_curve_data("P3/TrainSubsets")

train_error = []
test_error = []
models = []
percent_dim = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# for i in range(0, init_means.shape[0]):
#     cov = []
#     cov.append(np.identity(2))
#     cov.append(np.identity(2))
#     cov.append(np.identity(2))
#     cov = np.asarray(cov)
#     gmm = GaussianMixtureModel(3, init_means[i], cov)
#     gmm.fit(training_data[i])
#     datapoints = training_data[i].shape[0]
#     testpoints = test_data.shape[0]
#     train_err = gmm.compute_llh(training_data[i])
#     train_error.append(train_err)
#     test_err = gmm.compute_llh(test_data)
#     test_error.append(test_err)   
#     models.append(gmm)
#     print(gmm.covariances)
#     print(gmm.mus)
#     print(gmm.mixing_coeff)


# plot_multiple_contour_plots(models)
# plt.savefig('Model2.png')
# plt.show()

# print("FIRST ")
# print(train_error)
# print(test_error)

# trained_err = [-3.5233008836847177, -3.5732318537457775, -3.6301471234530376, -3.635997848247886, -3.6607199223977402, -3.675776730665362, -3.678851906737909, -3.683093561976469, -3.6953421909311888, -3.700142228271636]
# tested_err = [-11.168710512331034, -4.784168795335142, -3.1936031634995654, -2.410893320432663, -1.915094258869165, -1.5689085165805956, -1.3432435138525542, -1.1545744034004044, -1.0276602405146664, -0.9265581411732929]

trained_err = [-3.523302536291851, -3.573239087818134, -3.6301490085805335, -3.635999242607405, -3.6607231424298146, -3.675779446651391, -3.6788574797489266, -3.683506125664604, -3.6953586955208078, -3.7001608515067854]
tested_err = [-4.4673989261626295, -3.828914237852436, -3.831232446164829, -3.856596789411038, -3.8286039852490465, -3.7644217628823835, -3.7595495895898763, -3.722676569070129, -3.697497849559105, -3.70898860659202]

# plt.plot(percent_dim, tested_err, label='Test LLH')
# plt.plot(percent_dim, trained_err, label='Training LLH')
# plt.xlabel("Percent Data Used")
# plt.ylabel("Log Likelihood")
# plt.title("Training LLH + Test LLH vs Percent Data Used")
# plt.legend()
# plt.show()

# all_folds = load_all_cross_validation_data("P1/" + dataset + "/CrossValidation")

# gmm = GaussianMixtureModel(3, init_means[i], cov)

global_test = utils.load_data_from_txt_file("P3/X_test.txt")
global_train = utils.load_data_from_txt_file("P3/X_train.txt")

print("global_test", len(global_test), global_test.shape[0])

all_cvs = []
def cv_error(mean, cov, kval, cv_data):
    print("running CV on ", kval)
    average = 0
    gmm = None
    for i in range(0, 5):
        gmm = GaussianMixtureModel(kval, mean, cov)
        gmm.fit(cv_data[i][0])
#         datapoints = cv_data[i][1].shape[0]
#         print(datapoints)
        cv_err = gmm.compute_llh(cv_data[i][1])
        average = average + cv_err
        print(cv_err)
    gmm_test = GaussianMixtureModel(kval, mean, cov)
    gmm_test.fit(global_train)
    train_err = gmm_test.compute_llh(global_train)
    test_err = gmm_test.compute_llh(global_test)
    print(gmm_test.mixing_coeff)
    print(gmm_test.mus)
    print(gmm_test.covariances)

    average = average / (5)
    return average, train_err, test_err 


def generate_bmeans():
    part_bmean = []
    for num in range(1, 6): # For percent from 10, 20, ..., 100
        X_file = f"P3/MeanInitialization/Part_b/mu_k_{num}.txt"
        X = utils.load_data_from_txt_file(X_file)
        part_bmean.append(np.asarray(X))
    part_bmean = np.asarray(part_bmean)
    return part_bmean

def generate_initialcov(k):
    cov = []
    for i in range(0, k):
        cov.append(np.identity(2))
    return np.asarray(cov)

b_means = generate_bmeans()
partb_cv = load_all_cross_validation_data("P3/CrossValidation")


all_k_vals = [1, 2, 3, 4, 5]
all_cv_errs = []
all_test = []
all_train = []

# for i in range(0, 5):
#     k = i + 1
#     covs = generate_initialcov(k)
#     cv_err, train, test = cv_error(b_means[i], covs, k, partb_cv)
#     all_cv_errs.append(cv_err)
#     all_train.append(train)
#     all_test.append(test)

    
# print(all_cv_errs)
# print(all_test)
# print(all_train)
# print("All CV")
# print(all_cvs)

# plt.plot(all_k_vals, all_test, label='Test Error with Lambda')
# plt.plot(all_k_vals, all_train, label='Training Error with Lambda')
# plt.plot(all_k_vals, all_cv_errs, label='CV Error with Lambda')
# plt.xlabel("K Val Used")
# plt.ylabel("Log Likelihood")
# plt.title("Training Error, Test Error, CV Error vs K")
# plt.legend()
# plt.show()


cv_e = [-3.9832760054661733, -3.803344472406979, -3.7386511008828167, -3.754306524271894, -3.7575263062619606]
test_e = [-3.9432228329024057, -3.773904967516721, -3.7034163896733827, -3.7093865403663218, -3.7497747220351796]
train_e = [-3.977027741717642, -3.7813146950813588, -3.7001631973404367, -3.695419680283337, -3.675216502746491]


# cv_e = [-3.9832760054661733, -3.803189604963476, -3.738411115379491, -3.7552096109207227, -3.757495275300444]
# test_e = [-3.9432228329024057, -3.773643026857839, -3.7059700422254234, -3.708657379597118, -3.7501982911615874]
# train_e = [-3.977027741717642, -3.7813129278892785, -3.700142228028326, -3.692513429868624, -3.675190067733833]

plt.plot(all_k_vals, test_e, label='Test LLH')
plt.plot(all_k_vals, train_e, label='Training LLH')
plt.plot(all_k_vals, cv_e, label='CV Error')
plt.xlabel("K Val Used")
plt.ylabel("Log Likelihood")
plt.title("Training LLH, Test LLH, CV LLH vs K")
plt.legend()
plt.show()


# p = gmm.E_step(example_data)
# gmm.M_step(example_data, p)

# test1 = np.zeros(2)
# test2 = np.zeros(2).T
# print(test1.dot(test2))