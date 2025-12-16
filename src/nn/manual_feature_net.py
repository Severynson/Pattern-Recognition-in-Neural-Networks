import numpy as np
from .activations import relu, softmax

def build_handcrafted_feature_bank_vectors():
    # Each mask is already length-15 (row-major)
    right_edge = np.array([
        0,0,1,
        0,0,1,
        0,0,1,
        0,0,1,
        0,0,1
    ], dtype=np.float32)

    three_horizontal = np.array([
        1,1,1,
        0,0,0,
        1,1,1,
        0,0,0,
        1,1,1
    ], dtype=np.float32)

    middle_dot = np.array([
        0,0,0,
        0,0,0,
        0,1,0,
        0,0,0,
        0,0,0
    ], dtype=np.float32)

    row2_left = np.array([
        0,0,0,
        1,0,0,
        0,0,0,
        0,0,0,
        0,0,0
    ], dtype=np.float32)

    row2_right = np.array([
        0,0,0,
        0,0,1,
        0,0,0,
        0,0,0,
        0,0,0
    ], dtype=np.float32)

    row4_left = np.array([
        0,0,0,
        0,0,0,
        0,0,0,
        1,0,0,
        0,0,0
    ], dtype=np.float32)

    row4_right = np.array([
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,1,
        0,0,0
    ], dtype=np.float32)

    masks = [
        three_horizontal,
        middle_dot,
        right_edge,
        row2_left,
        row2_right,
        row4_left,
        row4_right,
    ]

    W_hand = np.stack(masks, axis=0)               # (7,15)
    b_hand = np.zeros(W_hand.shape[0], dtype=np.float32)  # (7,)
    return masks, W_hand, b_hand

import numpy as np

def build_manual_output_layer():
    """
    Output layer takes hidden vector h of length 7 and outputs 10 logits.
      z = W_out @ h + b_out
    Shapes:
      W_out: (10, 7)
      b_out: (10,)
    """

    # Each row is weights for one digit-class neuron over the 7 hidden features
    W_out = np.array([
        # class 0
        [  1, -100,   0,   1,   1,   1,   1],
        # class 1
        [ -1, -1000,  2, -1000, 0, -1000, 0],
        # class 2
        [  1,   0,    0,  -100, 0,   0,  -100],
        # class 3
        [  1,   0,    0,  -100, 1,  -100, 1],
        # class 4
        [  0,   1,    1,    3,  0,  -100, 0],
        # class 5
        [  1,   0,    0,    1, -100, -100, 1],
        # class 6
        [  1,   0,    0,    1, -100,   2,  1],
        # class 7
        [  0,   9,    0,  -100, 0,  -100, 0],
        # class 8
        [  1,   0,    0,    1,   1,   2,  1],
        # class 9
        [  1,   0,    0,    1,   1, -100, 1],
    ], dtype=np.float32)

    # One scalar bias per output neuron (class)
    # In your old pair format, only some classes had -1 somewhere; those summed into b.
    b_out = np.array([
        0,  # class 0
        0,  # class 1
        0,  # class 2
        0,  # class 3
        0,  # class 4
        0,  # class 5
        -1, # class 6
        0,  # class 7
        -1, # class 8
        0,  # class 9
    ], dtype=np.float32)

    return W_out, b_out


class ManualFeatureNet:
    def __init__(self):
        self.hidden_masks, self.W_hand, self.b_hand = build_handcrafted_feature_bank_vectors()
        self.W_out, self.b_out = build_manual_output_layer()

    def hidden_activations(self, x_vec):
        h_raw = self.W_hand @ x_vec + self.b_hand
        return relu(h_raw)

    def forward_probs(self, x_vec):
        h = self.hidden_activations(x_vec)
        z = self.W_out @ h + self.b_out

        return softmax(z)

    def predict(self, x_vec):
        return int(np.argmax(self.forward_probs(x_vec)))