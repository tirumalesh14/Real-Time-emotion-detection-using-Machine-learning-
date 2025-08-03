import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ Define emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ✅ Define dataset path
BASE_DIR = r"C:/Users/tirum/Downloads/archive"  # Update this with your dataset path
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# ✅ Data Augmentation for better training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 20% validation split
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ✅ Load training and validation dataset
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(48, 48), batch_size=64, color_mode="grayscale",
    class_mode="categorical", subset="training"
)

validation_generator = validation_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(48, 48), batch_size=64, color_mode="grayscale",
    class_mode="categorical", subset="validation"
)

# ✅ Function to create different CNN models
def build_model(model_type):
    if model_type == "lenet":
        model = models.Sequential([
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(6, (5, 5), activation='tanh'),
            layers.AveragePooling2D(pool_size=(2, 2)),  # ✅ Fixed
            layers.Conv2D(16, (5, 5), activation='tanh'),
            layers.AveragePooling2D(pool_size=(2, 2)),  # ✅ Fixed
            layers.Flatten(),
            layers.Dense(120, activation='tanh'),
            layers.Dense(84, activation='tanh'),
            layers.Dense(7, activation='softmax')
        ])

    elif model_type == "alexnet":
        model = models.Sequential([
            layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Conv2D(256, (5, 5), activation='relu', padding="same"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Conv2D(384, (3, 3), activation='relu', padding="same"),
            layers.Conv2D(384, (3, 3), activation='relu', padding="same"),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])

    elif model_type == "resnet":
        def residual_block(x, filters):
            shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)  # Ensure matching dimensions
            x = layers.Conv2D(filters, (3, 3), padding="same", activation='relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            return x

        inputs = layers.Input(shape=(48, 48, 1))
        x = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(inputs)
        x = residual_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)

        x = residual_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)

        x = residual_block(x, 256)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(7, activation='softmax')(x)

        model = models.Model(inputs, x)

    elif model_type == "googlenet":
        def inception_module(x, filters):
            path1 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
            path2 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
            path2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(path2)
            path3 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
            path3 = layers.Conv2D(filters, (5, 5), activation='relu', padding='same')(path3)
            path4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
            path4 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(path4)
            return layers.Concatenate()([path1, path2, path3, path4])

        inputs = layers.Input(shape=(48, 48, 1))
        x = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(inputs)
        x = inception_module(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)

        x = inception_module(x, 128)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(7, activation='softmax')(x)

        model = models.Model(inputs, x)

    elif model_type == "vgg":
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(48, 48, 1)),
            layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
            layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])

    return model


# ✅ Train and Save All Models
models_list = ["lenet", "alexnet", "resnet", "googlenet", "vgg"]

for model_type in models_list:
    print(f"🚀 Training model: {model_type}")

    model = build_model(model_type)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f"{model_type}_emotion_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,  # Adjust based on available hardware
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    print(f"✅ Model {model_type} trained and saved successfully!")

print("🎉 All models trained successfully!")
