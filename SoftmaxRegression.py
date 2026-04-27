
import numpy as np

class AdvancedSoftmaxRegression:
    def __init__(self, input_dim, num_classes, lr=0.1, reg=1e-4, momentum=0.9):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)

        self.lr = lr
        self.reg = reg
        self.momentum = momentum

        # momentum buffers
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        oh = np.zeros((len(y), num_classes))
        oh[np.arange(len(y)), y] = 1
        return oh

    def preprocess(self, X):
        return X  # DO NOTHING HERE

    def forward(self, X):
        return self.softmax(X @ self.W + self.b)

    def loss(self, y_pred, y_true):
        m = len(y_true)
        eps = 1e-9

        cross_entropy = -np.sum(y_true * np.log(y_pred + eps)) / m
        l2_penalty = self.reg * np.sum(self.W ** 2)

        return cross_entropy + l2_penalty

    def train(self, X, y, X_val, y_val, epochs=100, batch_size=32):
        X = self.preprocess(X)
        X_val = self.preprocess(X_val)

        num_classes = self.W.shape[1]
        y_oh = self.one_hot(y, num_classes)

        n = len(X)

        for epoch in range(epochs):
            # shuffle data
            idx = np.random.permutation(n)
            X, y_oh = X[idx], y_oh[idx]

            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_oh[i:i+batch_size]

                probs = self.forward(X_batch)

                error = probs - y_batch

                # gradients with L2 regularization
                dW = (X_batch.T @ error) / len(X_batch) + 2 * self.reg * self.W
                db = np.mean(error, axis=0)

                # momentum update
                self.vW = self.momentum * self.vW - self.lr * dW
                self.vb = self.momentum * self.vb - self.lr * db

                self.W += self.vW
                self.b += self.vb

            # learning rate decay
            self.lr *= 0.95

            # validation
            val_preds = self.predict(X_val)
            val_acc = np.mean(val_preds == y_val)

            if epoch % 10 == 0:
                train_loss = self.loss(self.forward(X), y_oh)
                print(f"Epoch {epoch}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    def predict(self, X):
        X = self.preprocess(X)
        probs = self.forward(X)
        return np.argmax(probs, axis=1)