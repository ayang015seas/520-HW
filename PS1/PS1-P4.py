import utils
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import LogisticRegression

# HINT: this is really all the imports you need
x_data, y_data = utils.load_learning_curve_data("P4/Train-subsets")
X_train, y_train, X_test, y_test = utils.load_all_train_test_data("P4")
folds = utils.load_all_cross_validation_data("P4/Cross-validation")

logistic = LogisticRegression(0)

alllam = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]

def lambdaError(lam, folds):
    average = 0
    logreg = LogisticRegression(lam)
    for i in range(0, 5):
        leave_out_data, training_data = utils.partition_cross_validation_fold(folds, i)
        logreg.fit(training_data[0], training_data[1])
        reg_pred = logreg.predict(leave_out_data[0])
        reg_err = utils.classification_error(reg_pred, leave_out_data[1])
        average = average + reg_err
    average = average / 5
    return average

lamerrs = []
lamclasserr = []
lamtesterr = []

for i in range(0, len(alllam)):
    logreg = LogisticRegression(alllam[i])
    logreg.fit(X_train, y_train)
    testpred = logreg.predict(X_test)
    trainpred = logreg.predict(X_train)
    class_error = utils.classification_error(testpred, y_test)
    training_error = utils.classification_error(trainpred, y_train)
    lamclasserr.append(class_error)
    lamtesterr.append(training_error)
    lerr = lambdaError(alllam[i], folds)
    print("Training Error: ", training_error, "|Test Error: ", class_error, "|Cross Validation Error: ", lerr, "|Lambda: ", alllam[i])
    lamerrs.append(lerr)

print(lamerrs)
    
training_cumulative = []
class_cumulative = []
datasize = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(0, len(x_data)):
    logistic.fit(x_data[i], y_data[i])
    class_pred = logistic.predict(X_test)
    training_pred = logistic.predict(x_data[i])
    class_error = utils.classification_error(class_pred, y_test)
    training_error = utils.classification_error(training_pred, y_data[i])
    print(class_error," ", training_error)
    class_cumulative.append(class_error)
    training_cumulative.append(training_error)

rtraining_cumulative = []
rclass_cumulative = []
logregoptimal = LogisticRegression(10**-3)
for i in range(0, len(x_data)):
    logregoptimal.fit(x_data[i], y_data[i])
    class_pred = logregoptimal.predict(X_test)
    training_pred = logregoptimal.predict(x_data[i])
    class_error = utils.classification_error(class_pred, y_test)
    training_error = utils.classification_error(training_pred, y_data[i])
    print(class_error," ", training_error)
    rclass_cumulative.append(class_error)
    rtraining_cumulative.append(training_error)

plt.plot(datasize, rclass_cumulative, label='Test Error with Lambda')
plt.plot(datasize, rtraining_cumulative, label='Training Error with Lambda')
plt.xlabel("Percent Data Used")
plt.ylabel("Percent Error")
plt.title("Training Error + Test Error vs Percent Data Used")
plt.legend()
plt.show()
    
plt.plot(alllam, lamerrs, label='Cross Validation Error')
plt.plot(alllam, lamclasserr, label='Test Error')
plt.plot(alllam, lamtesterr, label='Training Error')
plt.xlabel("Lambda Used")
plt.ylabel("Percent Error")
plt.xscale("log")
plt.title("Error vs Lambda Value Used")
plt.legend()
plt.show()


plt.plot(datasize, class_cumulative, label='Test Error')
plt.plot(datasize, training_cumulative, label='Training Error')
plt.xlabel("Percent Data Used")
plt.ylabel("Percent Error")
plt.title("Training Error + Test Error vs Percent Data Used")
plt.legend()
plt.show()


