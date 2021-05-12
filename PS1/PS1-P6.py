import utils
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import LeastSquareRegression

# Hint: this is all the imports you need
x_data, y_data = utils.load_learning_curve_data("P6/Data-set-1/Train-subsets")
X_train, y_train, X_test, y_test = utils.load_all_train_test_data("P6/Data-set-1")
# folds = utils.load_all_cross_validation_data("/Users/alexyang/desktop/ps1_kit/P6/Data-set-2/Cross-validation")


# custom data load
def load_data_custom(learning_curve_data_folder):
    """
    I do what I want
    """    
    all_X_train = []
    all_y_train = []
    
    for percent in range(10, 101, 90): # For percent from 10, 20, ..., 100
        X_file = f"{learning_curve_data_folder}/X_train_{percent}%.txt"
        y_file = f"{learning_curve_data_folder}/y_train_{percent}%.txt"
        X = utils.load_data_from_txt_file(X_file)
        y = utils.load_data_from_txt_file(y_file, True)
        all_X_train.append(X)
        all_y_train.append(y)
    
    return (all_X_train, all_y_train)

# custom cv load
def load_cv_data(validation_data_folder, percent):
    """
    custom cv data
    """
    all_folds = []
    for fold in [1,2,3,4,5]:
        X_file = f"{validation_data_folder}/Fold{fold}/X_{percent}%.txt"
        y_file = f"{validation_data_folder}/Fold{fold}/y_{percent}%.txt"
        X = utils.load_data_from_txt_file(X_file)
        y = utils.load_data_from_txt_file(y_file, True)
        all_folds.append((X,y))
    return all_folds


#load in 8 dimensional data
x_datareg, y_datareg = load_data_custom("P6/Data-set-2/Train-subsets")
X_train8, y_train8, X_test8, y_test8 = utils.load_all_train_test_data("P6/Data-set-2")

lin = LeastSquareRegression(0)


# part 1 with synthetic trainng data
training_cumulative = []
class_cumulative = []
datasize = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(0, len(x_data)):
    lin.fit(x_data[i], y_data[i])
    class_pred = lin.predict(X_test)
    training_pred = lin.predict(x_data[i])
    class_error = utils.mean_squared_error(class_pred, y_test)
    training_error = utils.mean_squared_error(training_pred, y_data[i])
    print(class_error," ", training_error)
    class_cumulative.append(class_error)
    training_cumulative.append(training_error)

total_pred = lin.predict(X_test)
print("W: ", lin.w, " b: ", lin.b)
plt.scatter(X_test, total_pred, label='Test Error')
plt.scatter(X_test, y_test, label='Training Error')
plt.legend()
plt.show()

plt.plot(datasize, class_cumulative, label='Test Error')
plt.plot(datasize, training_cumulative, label='Training Error')
plt.xlabel("Percent Data Used")
plt.ylabel("Percent Error")
plt.title("Training Error + Test Error vs Percent Data Used")
plt.legend()
plt.show()

# part 2 with 8 dimensionunregularized data

# print(len(x_datareg))
training_cumulative_unreg = []
class_cumulative_unreg = []
datasize8 = [10, 100]
for i in range(0, len(x_datareg)):
    lin.fit(x_datareg[i], y_datareg[i])
    class_pred = lin.predict(X_test8)
    training_pred = lin.predict(x_datareg[i])
    class_error = utils.mean_squared_error(class_pred, y_test8)
    training_error = utils.mean_squared_error(training_pred, y_datareg[i])
    print(datasize8[i], "Percent Used, Test Err: ", class_error," Train Err", training_error)
    print("W: ", lin.w)
    print("b: ", lin.b)
    class_cumulative_unreg.append(class_error)
    training_cumulative_unreg.append(training_error)

plt.plot(datasize8, class_cumulative_unreg, label='Test Error')
plt.plot(datasize8, training_cumulative_unreg, label='Training Error')
plt.xlabel("Percent Data Used")
plt.ylabel("Percent Error")
plt.title("Training Error + Test Error vs Percent Data Used")
plt.legend()
plt.show()


# handle 10 percent case first 
folds = load_cv_data("P6/Data-set-2/Cross-validation", 10)

alllam = [1**-1, 1, 10, 100, 500, 1000]
def lambdaError(lam, folds):
    average = 0
    linreg = LeastSquareRegression(lam)
    for i in range(0, 5):
        leave_out_data, training_data = utils.partition_cross_validation_fold(folds, i)
        linreg.fit(training_data[0], training_data[1])
        reg_pred = linreg.predict(leave_out_data[0])
        reg_err = utils.mean_squared_error(reg_pred, leave_out_data[1])
        average = average + reg_err
    average = average / 5
    return average

lamerrs10 = []
train_reg_err10 = []
class_reg_err10 = []

for i in range(0, len(alllam)):
    lerr = lambdaError(alllam[i], folds)
    linreg = LeastSquareRegression(alllam[i])
    linreg.fit(x_datareg[0], y_datareg[0])
    class_pred = linreg.predict(X_test8)
    training_pred = linreg.predict(x_datareg[0])
    class_error = utils.mean_squared_error(class_pred, y_test8)
    training_error = utils.mean_squared_error(training_pred, y_datareg[0])
    print("10% model, Test Err:", class_error," Train Err:", training_error)
    print("Lambda: ", alllam[i])
    print("W: ", linreg.w, "B: ", linreg.b)
    class_reg_err10.append(class_error)
    train_reg_err10.append(training_error)
    lamerrs10.append(lerr)

lamerrs100 = []
train_reg_err100 = []
class_reg_err100 = []

folds = load_cv_data("P6/Data-set-2/Cross-validation", 100)

for i in range(0, len(alllam)):
    lerr = lambdaError(alllam[i], folds)
    linreg = LeastSquareRegression(alllam[i])
    linreg.fit(x_datareg[1], y_datareg[1])
    class_pred = linreg.predict(X_test8)
    training_pred = linreg.predict(x_datareg[1])
    class_error = utils.mean_squared_error(class_pred, y_test8)
    training_error = utils.mean_squared_error(training_pred, y_datareg[1])
    print("100% model, Test Err:", class_error," Train Err:", training_error)
    print("Lambda: ", alllam[i])
    print("W: ", linreg.w, "B: ", linreg.b)
    class_reg_err100.append(class_error)
    train_reg_err100.append(training_error)
    lamerrs100.append(lerr)

print(alllam)
print(train_reg_err10)
print(class_reg_err10)
plt.plot(alllam, lamerrs10, label='Cross-Validation Error')
plt.plot(alllam, train_reg_err10, label='Train Error')
plt.plot(alllam, class_reg_err10, label='Class Error')
plt.xlabel("Lambda Used")
plt.ylabel("Percent Error 10 percent data")
plt.xscale("log")
plt.title("Error vs Lambda Value Used 10 percent")
plt.legend()
plt.show()

plt.plot(alllam, lamerrs100, label='Cross-Validation Error')
plt.plot(alllam, train_reg_err100, label='Train Error')
plt.plot(alllam, class_reg_err100, label='Class Error')
plt.xlabel("Lambda Used")
plt.ylabel("Percent Error 100 percent data")
plt.xscale("log")
plt.title("Error vs Lambda Value Used 100 percent")
plt.legend()
plt.show()

print(lamerrs100)