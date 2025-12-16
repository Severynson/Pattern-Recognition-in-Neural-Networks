import numpy as np
import matplotlib.pyplot as plt

def show_digit_vector(x_vec, height, width, title=None):
    mat = np.array(x_vec).reshape(height, width)
    fig = plt.figure(facecolor="black")
    plt.imshow(mat, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()

def show_filters_vectors(W, height, width, titles=None):
    """
    W: (k, input_size) where input_size = height*width
    """
    W = np.array(W)
    for i in range(W.shape[0]):
        t = None if titles is None else titles[i]
        show_digit_vector(W[i], height, width, title=t)
