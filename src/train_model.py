import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess_data import preprocess_data

def build_model(input_shape=(64, 64, 3), num_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_data, val_data, epochs=10, batch_size=32):
    model = build_model()
    model.fit(train_data, validation_data=val_data, epochs=epochs, batch_size=batch_size)
    return model

train_dir = 'data/train'
val_dir = 'data/val'
train_data, val_data = preprocess_data(train_dir, val_dir)
model = train_model(train_data, val_data, epochs=10)
