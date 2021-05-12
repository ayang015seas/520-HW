#PS2-P3.py starts here

import numpy as np 
import matplotlib.pyplot as plt
import utils
import time

from neural_network import NeuralNetworkClassification

folder = "Spam-Dataset"

X_train, y_train, X_test, y_test = utils.load_all_train_test_data("P3/" + folder)
folds = utils.load_all_cross_validation_data("P3/" + folder + "/CrossValidation")

instances = X_train.shape[0]
d = X_train.shape[1]
init_w = utils.load_initial_weights("P3/" + folder + "/InitParams/relu/10")

netlist = []

def cvError(dval, folds, mode, step):
    average = 0
    path = "P3/" + folder + "/InitParams/sigmoid/" + str(dval)
    initial_params = utils.load_initial_weights(path)

    for i in range(0, 5):
        nn = NeuralNetworkClassification(d, num_hidden=dval, activation=mode, W1=initial_params["W1"], W2=initial_params["W2"], b1=initial_params["b1"], b2=initial_params["b2"])
        leave_out_data, training_data = utils.partition_cross_validation_fold(folds, i)
        nn.fit(training_data[0], training_data[1], step_size=step)
        reg_pred = nn.predict(leave_out_data[0])
        reg_err = utils.classification_error(reg_pred, leave_out_data[1])
        average = average + reg_err
    average = average / 5
    return average

starttime = time.time()

nn = NeuralNetworkClassification(d, activation="relu", W1=init_w["W1"], W2=init_w["W2"], b1=init_w["b1"], b2=init_w["b2"])
# nn.forwardProp(X_train)
# nn.back_propagate(X_train, y_train)


# print(y_train.shape)


nn.fit(X_train, y_train)
endtime = time.time()
print(starttime - endtime)

# y_pred = nn.predict(X_train)
# print(y_pred)
# print(y_pred.shape)
# train_error = utils.classification_error(y_pred, y_train)
# y_test_pred = nn.predict(X_test)
# test_error = utils.classification_error(y_test_pred, y_test)
# print(train_error)
# print(test_error)

dvals = [1, 5, 10, 15, 25, 50]


def plotNeuralNetworks(mode, step_sz):
    lamerrs = []
    lamclasserr = []
    lamtesterr = []
    for i in range(0, len(dvals)):
        print("Starting CV ", i)
        path = "P3/" + folder + "/InitParams/sigmoid/" + str(dvals[i])
        initial_params = utils.load_initial_weights(path)
        nn = NeuralNetworkClassification(d, num_hidden=dvals[i], activation=mode, W1=initial_params["W1"], W2=initial_params["W2"], b1=initial_params["b1"], b2=initial_params["b2"])
        nn.fit(X_train, y_train, step_size=step_sz)
        testpred = nn.predict(X_test)
        trainpred = nn.predict(X_train)
        class_error = utils.classification_error(testpred, y_test)
        training_error = utils.classification_error(trainpred, y_train)
        print(class_error)
        print(training_error)
        lamclasserr.append(class_error)
        lamtesterr.append(training_error)
        lerr = cvError(dvals[i], folds, mode, step_sz)
    #     print("Training Error: ", training_error, "|Test Error: ", class_error, "|Cross Validation Error: ", lerr, "|Lambda: ", alllam[i])
        lamerrs.append(lerr)

    plt.plot(dvals, lamerrs, label='Cross Validation Error')
    plt.plot(dvals, lamclasserr, label='Test Error')
    plt.plot(dvals, lamtesterr, label='Training Error')
    print(lamtesterr)
    print(lamclasserr)
    print(lamerrs)
    plt.xlabel("C Used")
    plt.ylabel("Percent Error")
    plt.title("Error vs D Value Used")
    plt.legend()
    plt.show()

start = time.time()
plotNeuralNetworks("sigmoid", 0.12)
end = time.time()
print(end - start)

# Your code starts here.
