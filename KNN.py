import numpy as np
import os
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor

# ---------- CONFIG ----------
IMG_SIZE = (34, 34)
CACHE_X = "X.npy"
CACHE_Y = "y.npy"

# ---------- LABEL PARSER ----------
def get_label(filename):
    return int(filename.split('_')[0][1:])


# ---------- BORDER REMOVAL ----------
def remove_gray_border(img, tol=63):
    corners = [img[0,0], img[0,-1], img[-1,0], img[-1,-1]]
    bg = np.median(corners)

    mask = np.abs(img - bg) > tol
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    return img[y0:y1+1, x0:x1+1]


# ---------- PREPROCESS ----------
def preprocess_image(path, size=IMG_SIZE):
    # load grayscale
    img = Image.open(path).convert("L")
    img = np.array(img)

    # 1. remove gray border
    img = remove_gray_border(img, tol=63)

    # 2. resize (force fixed input shape)
    img = cv2.resize(img, size)

    # 3. normalize
    img = img.astype(np.float32) / 255.0

    return img


# ---------- SINGLE WORKER ----------
def load_one(args):
    folder, file = args
    path = os.path.join(folder, file)

    img = preprocess_image(path)
    label = get_label(file)

    return img, label


# ---------- DATASET LOADER ----------
def load_dataset(folder, use_cache=True):
    # ---- cache ----
    if use_cache and os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
        print("Loading from cache...")
        return np.load(CACHE_X), np.load(CACHE_Y)

    print("Loading from images...")

    files = sorted([f for f in os.listdir(folder) if f.endswith(".pgm")])

    # ---- parallel loading ----
    with ThreadPoolExecutor() as executor:
        data = list(executor.map(load_one, [(folder, f) for f in files]))

    X, y = zip(*data)

    X = np.stack(X)   # (N, 34, 34)
    y = np.array(y)

    # ---- cache ----
    if use_cache:
        np.save(CACHE_X, X)
        np.save(CACHE_Y, y)
        print("Saved cache.")

    return X, y


# ---------- SPLIT ----------
def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15):
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )


# ----------metrics----------
def l_1(x1, x2):
    return np.sum(np.abs(x1 - x2))

def l_2(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2)   )

# ---------- KNN ----------
class KNN:
    def __init__(self, k=3, metric='l2'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Flatten images: (N, H, W) -> (N, H*W)
        self.X_train = X.reshape(X.shape[0], -1)
        self.y_train = y

    def predict(self, X):
        # Flatten test images: (M, H, W) -> (M, H*W)
        X_test = X.reshape(X.shape[0], -1)
        
        predictions = []
        for x in X_test:
            # Compute distances using numpy vectorization
            if self.metric == 'l2':
                # Euclidean distance
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            else:
                # L1 distance
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            
            # Find k nearest neighbors using argpartition (O(N))
            k_indices = np.argpartition(distances, self.k)[:self.k]
            
            # Get labels and find the most common one
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
            
        return np.array(predictions)


if __name__ == "__main__":
    folder = r"C:/Users/TLP-001/PycharmProjects/DS_ML/For Students/Train Set (Unlabeled)"
    
    # Load and split dataset
    X, y = load_dataset(folder)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    print(f"Dataset loaded: {len(X)} samples")
    print(f"Training on {len(X_train)} samples...")
    
    # Initialize and fit KNN
    knn = KNN(k=3, metric='l1')
    knn.fit(X_train, y_train)
    
    # Predict on validation set
    print("Predicting on validation set...")
    val_preds = knn.predict(X_val)
    val_acc = np.mean(val_preds == y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Predict on test set
    print("Predicting on test set...")
    test_preds = knn.predict(X_test)
    test_acc = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")


