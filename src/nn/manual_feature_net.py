import numpy as np
from .activations import relu, softmax


def build_handcrafted_feature_bank():
    # Masks (5x3)
    right_edge = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    three_horizontal = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )

    middle_dot = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    row2_left = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    row2_right = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    row4_left = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    row4_right = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )

    hidden_masks = [
        three_horizontal,
        middle_dot,
        right_edge,
        row2_left,
        row2_right,
        row4_left,
        row4_right,
    ]
    W_hand = np.stack([m.reshape(-1) for m in hidden_masks], axis=0)  # (7, 15)
    b_hand = np.zeros(W_hand.shape[0], dtype=np.float32)
    return hidden_masks, W_hand.astype(np.float32), b_hand


def build_manual_output_layer():
    # each entry is 7 pairs [weight, bias]
    neuron_classifying_0 = [[1, 0], [-100, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    neuron_classifying_1 = [
        [-1, 0],
        [-1000, 0],
        [2, 0],
        [-1000, 0],
        [0, 0],
        [-1000, 0],
        [0, 0],
    ]
    neuron_classifying_2 = [
        [1, 0],
        [0, 0],
        [0, 0],
        [-100, 0],
        [0, 0],
        [0, 0],
        [-100, 0],
    ]
    neuron_classifying_3 = [
        [1, 0],
        [0, 0],
        [0, 0],
        [-100, 0],
        [1, 0],
        [-100, 0],
        [1, 0],
    ]
    neuron_classifying_4 = [[0, 0], [1, 0], [1, 0], [3, 0], [0, 0], [-100, 0], [0, 0]]
    neuron_classifying_5 = [
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [-100, 0],
        [-100, 0],
        [1, 0],
    ]
    neuron_classifying_6 = [[1, 0], [0, 0], [0, 0], [1, 0], [-100, 0], [2, -1], [1, 0]]
    neuron_classifying_7 = [
        [0, 0],
        [9, 0],
        [0, 0],
        [-100, 0],
        [0, 0],
        [-100, 0],
        [0, 0],
    ]
    neuron_classifying_8 = [[1, 0], [0, 0], [0, 0], [1, 0], [1, 0], [2, -1], [1, 0]]
    neuron_classifying_9 = [[1, 0], [0, 0], [0, 0], [1, 0], [1, 0], [-100, 0], [1, 0]]

    out_lists = [
        neuron_classifying_0,
        neuron_classifying_1,
        neuron_classifying_2,
        neuron_classifying_3,
        neuron_classifying_4,
        neuron_classifying_5,
        neuron_classifying_6,
        neuron_classifying_7,
        neuron_classifying_8,
        neuron_classifying_9,
    ]

    hidden_size = 7
    W_out = np.zeros((10, hidden_size), dtype=np.float32)
    b_out = np.zeros(10, dtype=np.float32)

    for k, neuron in enumerate(out_lists):
        neuron = np.array(neuron, dtype=np.float32)
        W_out[k] = neuron[:, 0]
        b_out[k] = neuron[:, 1].sum()

    return W_out, b_out


class ManualFeatureNet:
    def __init__(self):
        self.hidden_masks, self.W_hand, self.b_hand = build_handcrafted_feature_bank()
        self.W_out, self.b_out = build_manual_output_layer()

    def forward_probs(self, x_vec):
        h_raw = self.W_hand @ x_vec + self.b_hand
        h = relu(h_raw)
        z = self.W_out @ h + self.b_out
        return softmax(z)

    def predict(self, x_vec):
        return int(np.argmax(self.forward_probs(x_vec)))

    def hidden_activations(self, x_vec):
        h_raw = self.W_hand @ x_vec + self.b_hand
        return relu(h_raw)
