import numpy as np
from .activations import relu, softmax

def init_mlp(input_size, hidden_size, output_size, seed=42, w_scale=0.1):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, w_scale, size=(hidden_size, input_size)).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "W2": rng.normal(0, w_scale, size=(output_size, hidden_size)).astype(np.float32),
        "b2": np.zeros(output_size, dtype=np.float32),
    }

def forward_single(x, params):
    z1 = params["W1"] @ x + params["b1"]
    a1 = relu(z1)
    z2 = params["W2"] @ a1 + params["b2"]
    probs = softmax(z2)
    cache = {"x": x, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache

def predict_single(x, params):
    probs, _ = forward_single(x, params)
    return int(np.argmax(probs))
