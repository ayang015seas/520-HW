# # Principal component analysis

import numpy as np
import matplotlib.pyplot as plt
import utils
import math

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

# alldata = utils.load_data_from_txt_file("P1/X_train.csv")
alldata = np.genfromtxt("P1/X_train.csv", delimiter=',')
# target = alldata[3599]
# target = target.reshape(28, 28)
# plt.imshow(target, cmap="gray")
# plt.show()

## Your code starts here

def l2norm(row):
    squared = row * row
    rowsum = np.sum(squared)
#     print("rowsum", rowsum)
#     return (row / rowsum)
    return row / math.sqrt(rowsum)

testarray = np.arange(0, 6)
print(l2norm(testarray))
print(testarray)
    
def compute_mean(X):
    dims = X.shape
    total = np.sum(X, axis=0) / dims[0]
    print("Mean ", total.shape, dims[0])
    return total

def mean_adjust(X, mean):
    dims = X.shape
    mean_mat = np.tile(mean,(dims[0],1))
    print(mean_mat.shape)
    return X - mean_mat 

def k_eigen(X):
    U, E, V = np.linalg.svd(X)
    sliced = V
    # need to normalize 
    # 2norm of vector -> length of the vector -> 
    print("Sliced", len(sliced))
    for i in range(0, len(sliced)):
        sliced[i] = l2norm(sliced[i])
        
    normalized_array = sliced
    
    print("normal, ", normalized_array.shape)
    return normalized_array, E.flatten() * E.flatten()


# testmean = compute_mean(alldata)
# print(mean_adjust(alldata, testmean).shape)
# print(k_eigen(mean_adjust(alldata, compute_mean(alldata)), 3))
# score N x k 
# eigen is k x k

def reconstruct(score, eigenvec, mean):
    print(score.shape, eigenvec.shape)
    recon = score @ eigenvec
    recon = recon + mean
    return recon 

def reconstruct_error(X, reconstruction):
    diff = X - reconstruction
    diff = np.sum(diff, axis=0).flatten()
    return np.sum(diff)

def compute_score(X, k):
    mean = compute_mean(X)
    adjusted = mean_adjust(X, mean)
    eig, eigenvals = k_eigen(adjusted)
    
#     print(k_eig.shape)
    k_eig = eig[0:k] 
        
    score = adjusted @ k_eig.T
    print("Score Shape", score.shape)
    return score, eigenvals, k_eig, eig


component_num = []
sample_fraction = []

def reconstruction_accuracy(eigenvalues):
    print(eigenvalues)
    reconstruction_dict = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100 : 0}
    current_sum = 0
    total = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        
        if (current_sum / total < 0.10):
            reconstruction_dict[10] = i + 1
        elif (current_sum / total < 0.20):
            reconstruction_dict[20] = i + 1
        elif (current_sum / total < 0.30):
            reconstruction_dict[30] = i + 1
        elif (current_sum / total < 0.40):
            reconstruction_dict[40] = i + 1
        elif (current_sum / total < 0.50):
            reconstruction_dict[50] = i + 1
        elif (current_sum / total < 0.60):
            reconstruction_dict[60] = i + 1
        elif (current_sum / total < 0.70):
            reconstruction_dict[70] = i + 1
        elif (current_sum / total < 0.80):
            reconstruction_dict[80] = i + 1
        elif (current_sum / total < 0.90):
            reconstruction_dict[90] = i + 1
        current_sum = current_sum + eigenvalues[i]
        sample_fraction.append(current_sum / total)
        component_num.append(i)
        
    print(reconstruction_dict)
    return reconstruction_dict

def reconstructions(X, recon_dict, target):
    final_list = []
    mean = compute_mean(X)
    adjusted = mean_adjust(X, mean)
    eig, eigenvals = k_eigen(adjusted)
    
#     print(k_eig.shape)
    for i in range(10, 99, 10):
        print("iteration ", i)
        kvariable = recon_dict[i]
        k_eig = eig[0:kvariable]
        score = adjusted @ k_eig.T
        recon = reconstruct(score, k_eig, mean)
        row = recon[target].reshape(28, 28)
        print("row ", row.shape)
        final_list.append(row)
    final_list.append(X[target].reshape(28, 28))

    print("Recon Shape", np.asarray(final_list).shape)
    return np.asarray(final_list)
        
    print("Score Shape", score.shape)
    
    


mean = compute_mean(alldata)
score, eigenvals, eigenvecs, original_eigenvecs = compute_score(alldata, 700)

pc_1 = original_eigenvecs[0]
pc_2 = original_eigenvecs[1]
pc_3 = original_eigenvecs[2]

pc_1 = pc_1.reshape(28, 28)
pc_2 = pc_2.reshape(28, 28)
pc_3 = pc_3.reshape(28, 28)

# plt.imshow(pc_1, cmap="gray")
# plt.show()
# plt.imshow(pc_2, cmap="gray")
# plt.show()
# plt.imshow(pc_3, cmap="gray")
# plt.show()


X_adjusted = mean_adjust(alldata, mean)

dim_12 = eigenvecs[0: 2]
dim_101 = eigenvecs[99: 101]
low_dim = (X_adjusted @ dim_12.T).T
high_dim = (X_adjusted @ dim_101.T).T

print(low_dim)

# plt.scatter(low_dim[0], low_dim[1], s=1)
# plt.title("First and Second Dimension")
# plt.show()


# plt.scatter(high_dim[0], high_dim[1], s=1)
# plt.title("100 and 101 Dimension")
# plt.show()

recon = reconstruct(score, eigenvecs, mean)

target = recon[3599]
target = target.reshape(28, 28)

acc = reconstruction_accuracy(eigenvals)
print("SUM: ", np.sum(eigenvals))
print("Test: ", eigenvals[0] / np.sum(eigenvals))


recon_group = reconstructions(alldata, acc, 2999)
plot_reconstruction(recon_group, "3000th Example", "3000th")
plt.show()

recon_plot = plt.plot(component_num, sample_fraction, label='Accuracy')
plt.title("Reconstruction Accuracy")
plt.xlabel("Principal Components Used")
plt.ylabel("Reconstruction Accuracy")
plt.show()

# plt.imshow(target, cmap="gray")
# plt.show()

print(score)
