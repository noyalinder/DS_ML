from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
from SoftmaxRegression import AdvancedSoftmaxRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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


# ---------- EDGE FEATURES (IMPORTANT ADDITION) ----------
def extract_edges(img):
    img = img.astype(np.float32)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    return mag


# ---------- PREPROCESS ----------
def preprocess_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32)

    # 1. crop border
    img = remove_gray_border(img, tol=63)

    # 2. resize
    img = cv2.resize(img, size)

    # 3. per-image centering (IMPORTANT)
    img = img - np.mean(img)

    # 4. contrast normalization (IMPORTANT)
    img = img / (np.std(img) + 1e-8)

    # 5. edge features (IMPORTANT)
    edges = extract_edges(img)

    # normalize edges
    edges = edges / (np.max(edges) + 1e-8)

    # 6. combine features (still linear model!)
    combined = np.concatenate([img.flatten(), edges.flatten()])

    return combined


# ---------- SINGLE WORKER ----------
def load_one(args):
    folder, file = args
    path = os.path.join(folder, file)

    img = preprocess_image(path)
    label = get_label(file)

    return img, label


# ---------- DATASET LOADER ----------
def load_dataset(folder, use_cache=True):
    if use_cache and os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
        print("Loading from cache...")
        return np.load(CACHE_X), np.load(CACHE_Y)

    print("Loading from images...")

    files = sorted([f for f in os.listdir(folder) if f.endswith(".pgm")])

    with ThreadPoolExecutor() as executor:
        data = list(executor.map(load_one, [(folder, f) for f in files]))

    X, y = zip(*data)

    X = np.stack(X)
    y = np.array(y)

    if use_cache:
        np.save(CACHE_X, X)
        np.save(CACHE_Y, y)
        print("Saved cache.")

    return X, y


# ---------- SPLIT ----------
def split_dataset(X, y, train_ratio=0.7, val_ratio=0.01):
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


def augment_dataset(X, y, img_size=(34, 34), factor=1):
    h, w = img_size
    img_len = h * w

    X_aug = []
    y_aug = []

    for i in range(len(X)):
        img = X[i][:img_len].reshape(h, w)
        label = y[i]

        for _ in range(factor):
            # small transforms
            angle = np.random.uniform(-8, 8)
            tx = np.random.uniform(-2, 2)
            ty = np.random.uniform(-2, 2)

            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            M[:, 2] += [tx, ty]

            transformed = cv2.warpAffine(
                img, M, (w, h),
                borderMode=cv2.BORDER_REFLECT
            )

            # re-normalize
            transformed = transformed - np.mean(transformed)
            transformed = transformed / (np.std(transformed) + 1e-8)

            # recompute edges
            gx = cv2.Sobel(transformed, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(transformed, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(gx**2 + gy**2)
            edges = edges / (np.max(edges) + 1e-8)

            combined = np.concatenate([transformed.flatten(), edges.flatten()])

            X_aug.append(combined)
            y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

# ---------- LOAD ----------
folder = r"For Students\\Train Set (Labeled)"

X, y = load_dataset(folder)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

X_aug, y_aug = augment_dataset(X_train, y_train, factor=1)

# combine original + augmented
X_train = np.vstack([X_train, X_aug])
y_train = np.concatenate([y_train, y_aug])

print("finished loading")

# ---------- GLOBAL STANDARDIZATION (CRITICAL FIX) ----------
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val   = (X_val - mean) / std
X_test  = (X_test - mean) / std

X_train = X_train.reshape(len(X_train), -1)
X_val = X_val.reshape(len(X_val), -1)
X_test = X_test.reshape(len(X_test), -1)


print("feature std:", np.mean(np.std(X_train, axis=1)))



# ---------- MODEL ----------
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_leaf=2,
    min_samples_split=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# ---------- TRAIN ----------
rf_model.fit(X_train, y_train)

# ---------- EVALUATION ----------
train_preds = rf_model.predict(X_train)
test_preds = rf_model.predict(X_test)

train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)

print("Random Forest Training Accuracy:", train_acc)
print("Random Forest Testing Accuracy:", test_acc)

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()