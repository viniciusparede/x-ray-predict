from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import SparseCategoricalCrossentropy


class WallNet(Sequential):
    def __init__(self):
        super().__init__()

        # Adding the first convolutional layer
        self.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))

        # Adding a pooling layer to reduce dimensionality
        self.add(MaxPooling2D((2, 2)))

        # Adding a second convolutional layer
        self.add(Conv2D(64, (3, 3), activation="relu"))

        # Adding another pooling layer
        self.add(MaxPooling2D((2, 2)))

        # Adding a third convolutional layer
        self.add(Conv2D(64, (3, 3), activation="relu"))

        # Adding a Flatten layer to transform the feature matrix into a vector
        self.add(Flatten())

        # Adding a dense layer (or 'fully connected' layer)
        self.add(Dense(64, activation="relu"))

        # Adding the output layer
        self.add(Dense(10))

        # Compiling the model
        self.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

