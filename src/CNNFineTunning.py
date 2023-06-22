import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from SPRXRay import SPRXRay

# Load the SPR X-Ray dataset
data = SPRXRay()

(x_train, y_train), (x_test, y_test) = data.load_data(type_of_predict="gender")

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Load the ResNet50 model without the top layer (include_top=False)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation="relu")(x)

# Add a logistic layer with 10 classes (for CIFAR-10)
predictions = Dense(2, activation="softmax")(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model on the new data for a few epochs
model.fit(x_train, y_train, epochs=5)

# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from ResNet50. We will freeze the bottom N layers
# and train the remaining top layers.

# Let's visualize layer names and layer indices to see how many layers we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# We chose to train the top 2 ResNet blocks, i.e. we will freeze
# the first 165 layers and unfreeze the rest:
for layer in model.layers[:165]:
    layer.trainable = False
for layer in model.layers[165:]:
    layer.trainable = True

# We need to recompile the model for these modifications to take effect
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# We train our model again (this time fine-tuning the top 2 ResNet blocks)
# alongside the top Dense layers
model.fit(x_train, y_train, epochs=5)
