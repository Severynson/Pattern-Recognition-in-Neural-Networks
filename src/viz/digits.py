import numpy as np
import matplotlib.pyplot as plt


def show_digit(digit_matrix, title=None):
    fig = plt.figure(facecolor="black")
    plt.imshow(digit_matrix, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def show_digit_vector(x_vec, height, width, title=None):
    mat = np.array(x_vec).reshape(height, width)
    show_digit(mat, title=title)


def show_filters(filters, height, width, titles=None):
    """
    filters: shape (k, height*width) or (k, height, width)
    """
    arr = np.array(filters)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], height, width)

    for i in range(arr.shape[0]):
        t = None if titles is None else titles[i]
        show_digit(arr[i], title=t)
