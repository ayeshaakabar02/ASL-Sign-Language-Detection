import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc

# === Config ===
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 28
TRAIN_DIR = 'data/train'    # ✅ Updated
TEST_DIR = 'data/test'      # ✅ Updated
SEED = 42

# === Data Generators ===
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   zoom_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

# === Build Model Function ===
def build_model(base_model):
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

# === Model Dictionary ===
model_dict = {
    "mobilenetv2": MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    "nasnetmobile": NASNetMobile(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
}

# === Train & Evaluate Each Model ===
for name, base_model in model_dict.items():
    print(f"\n📦 Training Model: {name.upper()}")
    model = build_model(base_model)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # === Callbacks ===
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(f"{name}_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

    # === Training ===
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # === Final Evaluation ===
    loss, acc = model.evaluate(test_data, verbose=0)
    print(f"✅ Final Accuracy for {name.upper()}: {acc * 100:.2f}%")

    # === Save Final Model (even if not best) ===
    model.save(f"{name}_final.h5")
    print(f"💾 Final model saved as {name}_final.h5")

    # === Cleanup ===
    tf.keras.backend.clear_session()
    gc.collect()
