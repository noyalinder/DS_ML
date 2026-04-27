import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
from SoftmaxRegression import AdvancedSoftmaxRegression


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


folder = r"C:\Users\TLP-001\PycharmProjects\DS_ML-1\For Students\Train Set (Labeled)"

X, y = load_dataset(folder)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

print("finished loading")
print(X_train[0])

model = AdvancedSoftmaxRegression(
    input_dim=34*34,
    num_classes=28,
    lr=0.5,        # lower than 2 (2 is way too high now)
    reg=1e-4,      # L2 regularization strength
    momentum=0.9   # helps convergence
)

model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=200,
    batch_size=64   # new parameter
)

preds = model.predict(X_train)
acc = np.mean(preds == y_train)

print("Training accuracy:", acc)

preds = model.predict(X_test)
acc = np.mean(preds == y_test)

print("Testing accuracy:", acc)