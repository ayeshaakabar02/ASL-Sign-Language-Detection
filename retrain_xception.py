import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Config ===
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 28
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# === Data Generators ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Load Pretrained Xception Model ===
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# ✅ Freeze all, unfreeze last 30 layers only
base_model.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# === Custom Top Layers ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("xception_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# === Train with limited steps (faster) ===
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    steps_per_epoch=200,
    validation_steps=50,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# === Save final model (last epoch) ===
model.save("xception_final.h5")
print("✅ Model saved as xception_best.h5 and xception_final.h5")
