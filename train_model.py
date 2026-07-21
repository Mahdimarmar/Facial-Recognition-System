import numpy as np
import os
import cv2
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.utils import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# CONFIG

DATASET_PATH    = "utkcropped"
IMG_SIZE        = 64
N_JOBS          = -1
PCA_COMPONENTS  = 150
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 10
DL_EPOCHS       = 50    # increased from 30 — EarlyStopping handles it
DL_BATCH        = 64


# 1. FAST IMAGE LOADING  (parallel with joblib)


def load_image(image, dataset_path, img_size):
    """Load one image and return (pixels, age, gender, race) or None."""
    if not image.endswith('.jpg'):
        return None
    parts = image.split("_")
    if len(parts) < 3:
        return None
    if not (parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()):
        return None

    age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])

    img_path = os.path.join(dataset_path, image)
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (img_size, img_size))
    return img, age, gender, race


print("Loading images in parallel …")
files   = os.listdir(DATASET_PATH)
results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(load_image)(f, DATASET_PATH, IMG_SIZE) for f in files
)
results = [r for r in results if r is not None]

faces   = np.array([r[0] for r in results], dtype=np.float32) / 255.0
ages    = np.array([r[1] for r in results])
genders = np.array([r[2] for r in results])
races   = np.array([r[3] for r in results])

print(f"Loaded {len(faces)} images.")


# 2. TRAIN / TEST SPLIT  (shared by both ML and DL)

x_train, x_test, age_train, age_test, gender_train, gender_test, race_train, race_test = \
    train_test_split(faces, ages, genders, races,
                     test_size=0.2, random_state=1)


# 3. ML MODEL  — Random Forest via PCA pipeline

print("\n--- Training ML models ---")

x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat  = x_test.reshape(len(x_test),  -1)

# ---- Gender classifier ----
rf_gender = Pipeline([
    ('pca', PCA(n_components=PCA_COMPONENTS, random_state=42)),
    ('rf',  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                   criterion='gini', n_jobs=N_JOBS, random_state=42))
])
rf_gender.fit(x_train_flat, gender_train)
print("Gender model trained.")

# ---- Race classifier (class_weight='balanced' fixes imbalance) ----
rf_race = Pipeline([
    ('pca', PCA(n_components=PCA_COMPONENTS, random_state=42)),
    ('rf',  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                   criterion='gini', class_weight='balanced',
                                   n_jobs=N_JOBS, random_state=42))
])
rf_race.fit(x_train_flat, race_train)
print("Race model trained.")

# ---- Age regressor ----
rf_age = Pipeline([
    ('pca', PCA(n_components=PCA_COMPONENTS, random_state=42)),
    ('rf',  RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                  criterion='squared_error', n_jobs=N_JOBS, random_state=42))
])
rf_age.fit(x_train_flat, age_train)
print("Age model trained.")

# ---- ML Evaluation ----
ml_gender_acc = accuracy_score(gender_test, rf_gender.predict(x_test_flat))
ml_race_acc   = accuracy_score(race_test,   rf_race.predict(x_test_flat))
ml_age_mae    = mean_absolute_error(age_test, rf_age.predict(x_test_flat))

print(f"\n[ML Results]")
print(f"  Gender Accuracy : {ml_gender_acc:.4f}")
print(f"  Race   Accuracy : {ml_race_acc:.4f}")
print(f"  Age    MAE      : {ml_age_mae:.2f} years")

# ==============================================================
# 4. DATA AUGMENTATION SETUP
#    datagen.flow() doesn't support dict labels so we use a
#    custom generator that augments images and yields dict labels
# ==============================================================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

def augmented_generator(x, y_gender, y_race, y_age, batch_size):
    """Yields augmented batches with dictionary labels."""
    n = len(x)
    while True:
        indices = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx      = indices[start:start + batch_size]
            x_batch        = x[batch_idx]
            y_gender_batch = y_gender[batch_idx]
            y_race_batch   = y_race[batch_idx]
            y_age_batch    = y_age[batch_idx]

            # Apply augmentation to each image in the batch
            x_aug = np.stack([
                datagen.random_transform(img) for img in x_batch
            ])

            yield x_aug, {
                'gender_output': y_gender_batch,
                'race_output':   y_race_batch,
                'age_output':    y_age_batch
            }


# 5. DL MODEL  — Multi-output CNN

print("\n--- Building CNN model ---")

inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
race_output   = Dense(5, activation='softmax', name='race_output')(x)
age_output    = Dense(1, activation='linear',  name='age_output')(x)

model = Model(inputs=inp, outputs=[gender_output, race_output, age_output])


# Weighted race loss — fixes imbalance for multi-output model

race_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(race_train),
    y=race_train
)
race_weight_tensor = tf.constant(race_class_weights, dtype=tf.float32)

def weighted_race_loss(y_true, y_pred):
    """Sparse categorical crossentropy with per-class weights baked in."""
    y_true  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    weights = tf.gather(race_weight_tensor, y_true)
    loss    = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss * weights)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'gender_output': 'binary_crossentropy',
        'race_output':   weighted_race_loss,
        'age_output':    'mae'
    },
    loss_weights={
        'gender_output': 1.0,
        'race_output':   1.0,
        'age_output':    0.05
    },
    metrics={
        'gender_output': 'accuracy',
        'race_output':   'accuracy',
        'age_output':    'mae'
    }
)

model.summary()

# Callbacks

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'face_model_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Train

print("\n--- Training CNN model ---")

steps_per_epoch = len(x_train) // DL_BATCH

history = model.fit(
    augmented_generator(x_train, gender_train, race_train, age_train, DL_BATCH),
    steps_per_epoch=steps_per_epoch,
    validation_data=(
        x_test,
        {
            'gender_output': gender_test,
            'race_output':   race_test,
            'age_output':    age_test
        }
    ),
    epochs=DL_EPOCHS,
    callbacks=callbacks
)


# Evaluate

gender_pred_dl, race_pred_dl, age_pred_dl = model.predict(x_test, batch_size=DL_BATCH)

gender_pred_dl = (gender_pred_dl > 0.5).astype(int).flatten()
race_pred_dl   = np.argmax(race_pred_dl, axis=1)
age_pred_dl    = age_pred_dl.flatten()

dl_gender_acc = accuracy_score(gender_test, gender_pred_dl)
dl_race_acc   = accuracy_score(race_test,   race_pred_dl)
dl_age_mae    = mean_absolute_error(age_test, age_pred_dl)

print(f"\n[DL Results]")
print(f"  Gender Accuracy : {dl_gender_acc:.4f}")
print(f"  Race   Accuracy : {dl_race_acc:.4f}")
print(f"  Age    MAE      : {dl_age_mae:.2f} years")

print("\n========== COMPARISON ==========")
print(f"{'Metric':<25} {'ML (RF)':>12} {'DL (CNN)':>12}")
print("-" * 51)
print(f"{'Gender Accuracy':<25} {ml_gender_acc:>12.4f} {dl_gender_acc:>12.4f}")
print(f"{'Race Accuracy':<25} {ml_race_acc:>12.4f} {dl_race_acc:>12.4f}")
print(f"{'Age MAE (years)':<25} {ml_age_mae:>12.2f} {dl_age_mae:>12.2f}")

model.save("face_model.keras")
print("\nModel saved to face_model.keras (best weights in face_model_best.keras)")