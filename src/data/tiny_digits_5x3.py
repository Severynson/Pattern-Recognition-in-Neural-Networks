import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH = 5, 3
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH  # 15

def load_tiny_digits_vectors():
    # Images of digits represented as vectors of length 15
    X = np.array([
        # 0
        [
            1,1,1,
            1,0,1,
            1,0,1,
            1,0,1,
            1,1,1
         ],
        # 1
        [
            0,0,1,
            0,0,1,
            0,0,1,
            0,0,1,
            0,0,1
        ],
        # 2
        [
            1,1,1,
            0,0,1,
            1,1,1,
            1,0,0,
            1,1,1
        ],
        # 3
        [
            1,1,1,
            0,0,1,
            1,1,1,
            0,0,1,
            1,1,1
        ],
        # 4
        [
            1,0,1,
            1,0,1,
            1,1,1,
            0,0,1,
            0,0,1
        ],
        # 5
        [
            1,1,1,
            1,0,0,
            1,1,1,
            0,0,1,
            1,1,1
        ],
        # 6
        [
            1,1,1,
            1,0,0,
            1,1,1,
            1,0,1,
            1,1,1
        ],
        # 7
        [
            1,1,1,
            0,0,1,
            0,1,1,
            0,0,1,
            0,0,1
        ],
        # 8
        [
            1,1,1,
            1,0,1,
            1,1,1,
            1,0,1,
            1,1,1
        ],
        # 9
        [
            1,1,1,
            1,0,1,
            1,1,1,
            0,0,1,
            1,1,1
        ],
    ], dtype=np.float32)

    y = np.arange(10, dtype=np.int64)
    meta = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH, "input_size": INPUT_SIZE}
    return X, y, meta
