import numpy as np
from .mlp import forward_single


def cross_entropy_loss(probs, true_label, eps=1e-12):
    # We are in single-label classification (only 1 correct class with  100% prob. of a single correct class, 0% for others).
    # While general cross-entropy definition is:
    #   L = - ∑_i ( true_prob_i * log(predicted_prob_i) )

    # Example:
    #   Classes: [a, b, c]
    #   True probabilities:      y = [1, 0, 0]      (class 'a' is correct)
    #   Predicted probabilities: p = [0.33, 0.33, 0.34]

    # Plug into the formula:
    #   L = - (1 * log(0.33) + 0 * log(0.33) + 0 * log(0.34))
    #     = - log(0.33)

    # All terms except the correct class vanish because their true probability is 0.
    # So cross-entropy collapses to:
    #   L = -log(predicted probability assigned by the model to the true class)

    # probs[true_label] picks p_correct from the predicted distribution.
    # eps is added only to avoid log(0) for numerical stability.

    return -np.log(probs[true_label] + eps)


def train_mlp(X, y, params, learning_rate=0.1, epochs=1000, print_every=100):
    n = X.shape[0]

    for epoch in range(1, epochs + 1):
        dW1 = np.zeros_like(params["W1"])
        db1 = np.zeros_like(params["b1"])
        dW2 = np.zeros_like(params["W2"])
        db2 = np.zeros_like(params["b2"])

        total_loss = 0.0
        correct = 0

        for i in range(n):
            x_i = X[i]
            y_i = y[i]

            probs, cache = forward_single(x_i, params)
            total_loss += cross_entropy_loss(probs, y_i)
            if int(np.argmax(probs)) == int(y_i):
                correct += 1

            # dL/dz2
            dz2 = probs.copy()
            dz2[y_i] -= 1.0

            a1 = cache["a1"]
            dW2 += np.outer(dz2, a1)
            db2 += dz2

            # backprop to hidden
            dz1 = params["W2"].T @ dz2
            dz1[cache["z1"] <= 0] = 0

            dW1 += np.outer(dz1, cache["x"])
            db1 += dz1

        # average
        dW1 /= n
        db1 /= n
        dW2 /= n
        db2 /= n

        # step
        params["W1"] -= learning_rate * dW1
        params["b1"] -= learning_rate * db1
        params["W2"] -= learning_rate * dW2
        params["b2"] -= learning_rate * db2

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"[MLP] Epoch {epoch:4d} | loss={total_loss/n:.4f} | acc={100*correct/n:.1f}%"
            )

    return params
