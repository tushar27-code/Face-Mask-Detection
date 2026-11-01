
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
mymodel = Sequential()

mymodel.add(Conv2D(32, (3, 3), activation='relu',
                   input_shape=(150, 150, 3)))
mymodel.add(MaxPooling2D(pool_size=(2, 2)))

mymodel.add(Conv2D(32, (3, 3), activation='relu'))
mymodel.add(MaxPooling2D(pool_size=(2, 2)))

mymodel.add(Conv2D(32, (3, 3), activation='relu'))
mymodel.add(MaxPooling2D(pool_size=(2, 2)))

mymodel.add(Flatten())
mymodel.add(Dense(100, activation='relu'))
mymodel.add(Dense(1, activation='sigmoid'))

# Compile the model
mymodel.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Prepare the training and testing data
train = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test = ImageDataGenerator(rescale=1.0 / 255)

train_img = train.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_img = test.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Train the model
mask_model = mymodel.fit(
    train_img,
    epochs=10,
    validation_data=test_img
)

# Save the trained model
mymodel.save('mask.keras')
print("Class Mapping:", train_img.class_indices)

print("Model trained and saved successfully!")
