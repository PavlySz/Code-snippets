'''For when you need to quickly visualize random samples from your dataset'''
import numpy as np
import matplotlib.pyplot as plt


def plot_n_images_with_labels(x, y, n=25, one_hot=False):
    '''
    Plot n images alongside their labels.
    You may need to adjust the fgure size.

    Args:
        - x: set of images (may be train or test set)
        - y: labels corresponding to x
        - n: number of images to plot
        - one_hot: whether the labels are one-hot encoded or not
    '''
    # Generate random integers to act as indices
    rand_ind = np.random.randint(0, len(x), n)

    # Makes the axes
    _, axes = plt.subplots(5, 5, figsize=(10, 12))

    # Get images
    images = [x[index].reshape(28, 28) for index in rand_ind]

    # Get labels
    # If the labels are one-hot encoded, get the argmax of each one
    if one_hot:
        labels = [np.argmax(y[index]) for index in rand_ind]
    else:
        labels = [y[index] for index in rand_ind]

    # Iterate over zip(flattened axes, images and labels)
    # and plot each image alongside its label on an axis
    for ax, img, lbl in zip(axes.ravel(), images, labels):
        ax.imshow(img, cmap=plt.cm.get_cmap('gray'))
        ax.set_title(lbl, color='w', size=16)
        ax.set_xticks([])
        ax.set_yticks([])
