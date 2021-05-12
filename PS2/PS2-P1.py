# PS2-P1.py starts here

import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import math

from support_vector_machines import SVM

# Helper functions to draw decision boundary plot
def plot_contours(clf, X, y, n=100):
    """
    Produce classification decision boundary

    Args:
        clf:
            Any classifier object that predicts {-1, +1} labels
        
        X (numpy.array):
            A 2d feature matrix

        y (numpy.array):
            A {-1, +1} label vector

        n (int)
            Number of points to partition the meshgrids
            Default = 100.

    Returns:
        (fig, ax)
            fig is the figure handle
            ax is the single axis in the figure

        One can use fig to save the figure.
        Or ax to modify the title/axis label etc

    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]

    # Set-up grid for plotting.
    xx, yy = np.meshgrid(np.linspace(X0.min()-1, X0.max()+1, n),\
                         np.linspace(X1.min()-1, X1.max()+1, n),\
                        )
    # Do prediction for every single point on the mesh grid
    # This will take a few seconds
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["cyan", "pink"]))

    # Scatter the -1 points
    ax.scatter([X0[i] for i,v in enumerate(y) if v == -1],
                        [X1[i] for i,v in enumerate(y) if v == -1], 
                        c="blue", label='- 1',
                        marker='x')
    # Scatter the +1 points
    ax.scatter([X0[i] for i,v in enumerate(y) if v == 1],
                        [X1[i] for i,v in enumerate(y) if v == 1], 
                        edgecolor="red", label='+1', facecolors='none', s=10,
                        marker='o')

    ax.set_ylabel('x_2')
    ax.set_xlabel('x_1')
    ax.legend()
    return fig, ax

dataset = "Spam-Dataset"

# Your code starts here.
X_train, y_train, X_test, y_test = utils.load_all_train_test_data("P1/" + dataset)
folds_1a = utils.load_all_cross_validation_data("P1/" + dataset + "/CrossValidation")



def lambdaError(kernel, cval, folds):
    average = 0
    failedTrains = 0
    svm = SVM(kernel, cval)
    for i in range(0, 5):
        leave_out_data, training_data = utils.partition_cross_validation_fold(folds, i)
        status = svm.fit(training_data[0], training_data[1])
        reg_pred = svm.predict(leave_out_data[0])
        reg_err = utils.classification_error(reg_pred, leave_out_data[1])
        average = average + reg_err
    average = average / (5 - failedTrains)
    return average


cval_1a = [10**-4, 10**-3, 10**-2, 10**-1, 1, 10**1, 10**2]


# set up 1a - 1

def linearSVMPlot():
    lamerrs = []
    lamclasserr = []
    lamtesterr = []
    for i in range(0, len(cval_1a)):
        print(i)
        svm = SVM(None, cval_1a[i])
        svm.fit(X_train, y_train)
        testpred = svm.predict(X_test)
        trainpred = svm.predict(X_train)
        class_error = utils.classification_error(testpred, y_test)
        training_error = utils.classification_error(trainpred, y_train)
        lamclasserr.append(class_error)
        lamtesterr.append(training_error)
        lerr = lambdaError(None, cval_1a[i], folds_1a)
    #     print("Training Error: ", training_error, "|Test Error: ", class_error, "|Cross Validation Error: ", lerr, "|Lambda: ", alllam[i])
        lamerrs.append(lerr)

    plt.plot(cval_1a, lamerrs, label='Cross Validation Error')
    plt.plot(cval_1a, lamclasserr, label='Test Error')
    plt.plot(cval_1a, lamtesterr, label='Training Error')
    print(lamtesterr)
    print(lamclasserr)
    print(lamerrs)
    plt.xlabel("C Used")
    plt.ylabel("Percent Error")
    plt.xscale("log")
    plt.title("Error vs C Value Used")
    plt.legend()
    plt.show()

# linearSVMPlot()

# plot linear
# test_svm = SVM(None, 10**-1)
# test_svm.fit(X_train, y_train)
# plot_contours(test_svm, X_test, y_test)

# attempt 2 plot quad 
# quad = lambda x1, x2: (np.dot(x1, x2) + 1)**3
# test_svm2 = SVM(quad, 0.0001)
# test_svm2.fit(X_train, y_train)
# plot_contours(test_svm2, X_test, y_test)
# plt.show()

# rbflist = [10**-2, 10**-1, 10**0, 10**1, 10**2]
# rbf = lambda x1, x2: math.exp((x1 - x2).dot(x1 - x2) * -10**-1)
# test_svm2 = SVM(rbf, 0.1)
# test_svm2.fit(X_train, y_train)
# plot_contours(test_svm2, X_test, y_test)
# plt.show()

# set up 1a - 2 
def trainQuadratic(cvals, kernel):
    lamerrs = []
    lamclasserr = []
    lamtesterr = []
    
    min_error = 100
    c_choice = -1
    for i in range(0, len(cvals)):
        print(i)
        svm = SVM(kernel, cvals[i])
        svm.fit(X_train, y_train)
        testpred = svm.predict(X_test)
        trainpred = svm.predict(X_train)
        class_error = utils.classification_error(testpred, y_test)
        training_error = utils.classification_error(trainpred, y_train)
        lamclasserr.append(class_error)
        lamtesterr.append(training_error)
        lerr = lambdaError(kernel, cvals[i], folds_1a)
        if (lerr < min_error):
            min_error = lerr
            c_choice = cvals[i]
#         print("Training Error: ", training_error, "|Test Error: ", class_error, "|Cross Validation Error: ", lerr, "|Lambda: ", alllam[i])
        lamerrs.append(lerr)

    plt.plot(cvals, lamerrs, label='Cross Validation Error')
    plt.plot(cvals, lamclasserr, label='Test Error')
    plt.plot(cvals, lamtesterr, label='Training Error')
    plt.xlabel("C Used")
    plt.ylabel("Percent Error")
    plt.xscale("log")
    plt.title("Error vs C Value Used")
    plt.legend()
    plt.show()
    return c_choice

# trainQuadratic(cval_1a, None)
# = lambda x1, x2: np.dot(x1, x2)

quadlist = [1, 2, 3, 4, 5]
optimal_c = []
# for i in range(0, len(quadlist)):
#     opt_c = trainQuadratic(cval_1a, lambda x1, x2: (np.dot(x1, x2) + 1)**quadlist[i])
#     optimal_c.append(opt_c)
# print(optimal_c)


def plotPolyOptimal(cvals, qvals):
    lamerrs = []
    lamclasserr = []
    lamtesterr = []
    
    for i in range(0, len(qvals)):
        print(i)
        svm = SVM(lambda x1, x2: (np.dot(x1, x2) + 1)**qvals[i], cvals[i])
        svm.fit(X_train, y_train)
        testpred = svm.predict(X_test)
        trainpred = svm.predict(X_train)
        class_error = utils.classification_error(testpred, y_test)
        training_error = utils.classification_error(trainpred, y_train)
        lamclasserr.append(class_error)
        lamtesterr.append(training_error)
        lerr = lambdaError(lambda x1, x2: (np.dot(x1, x2) + 1)**qvals[i], cvals[i], folds_1a)
        lamerrs.append(lerr)
    print(lamtesterr)
    print(lamclasserr)
    print(lamerrs)
    plt.plot(qvals, lamerrs, label='Cross Validation Error')
    plt.plot(qvals, lamclasserr, label='Test Error')
    plt.plot(qvals, lamtesterr, label='Training Error')
    plt.xlabel("Q Used")
    plt.ylabel("Percent Error")
    plt.title("Error vs Q Value Used")
    plt.legend()
    plt.show()

def plotRBF(cvals, yvals):
    lamerrs = []
    lamclasserr = []
    lamtesterr = []
    
    rbf_eq = lambda x1, x2: math.exp((x1 - x2).dot(x1 - x2) * -yvals[i])
    for i in range(0, len(yvals)):
        print(i)
        svm = SVM(rbf_eq, cvals[i])
        svm.fit(X_train, y_train)
        testpred = svm.predict(X_test)
        trainpred = svm.predict(X_train)
        class_error = utils.classification_error(testpred, y_test)
        training_error = utils.classification_error(trainpred, y_train)
        lamclasserr.append(class_error)
        lamtesterr.append(training_error)
        lerr = lambdaError(rbf_eq, cvals[i], folds_1a)
        lamerrs.append(lerr)
    print(lamtesterr)
    print(lamclasserr)
    print(lamerrs)

    plt.plot(yvals, lamerrs, label='Cross Validation Error')
    plt.plot(yvals, lamclasserr, label='Test Error')
    plt.plot(yvals, lamtesterr, label='Training Error')
    plt.xlabel("y Used")
    plt.xscale("log")
    plt.ylabel("Percent Error")
    plt.title("Error vs y Value Used")
    plt.legend()
    plt.show()
    
bestC = [0.01, 0.01, 0.0001, 0.0001, 0.0001]
bestSpamC = [100, 10, 100, 100, 10]
# plotPolyOptimal(bestSpamC, quadlist)
# plotPolyOptimal(bestC, quadlist)


cval_1a3 = [10**-3, 10**-2, 10**-1, 1, 10**1, 10**2]

# trainQuadratic(cval_1a3, lambda x1, x2: math.exp(math.sqrt((x1 - x2).dot(x1 - x2)) * yvals[i]))
rbflist = [10**-2, 10**-1, 10**0, 10**1, 10**2]
optimal_c_rbf = []
# rbf_equation = lambda x1, x2: math.exp(((x1 - x2).dot(x1 - x2)) * rbflist[i])
# for i in range(0, len(rbflist)):
#     opt_c_rbf = trainQuadratic(cval_1a3, lambda x1, x2: math.exp((x1 - x2).dot(x1 - x2) * -rbflist[i]))
#     optimal_c_rbf.append(opt_c_rbf)
print(optimal_c_rbf)

# best_crbf = [1, 0.1, 1, 1, 10]
best_crbf = [100, 100, 100, 10, 1]
plotRBF(best_crbf, rbflist)

