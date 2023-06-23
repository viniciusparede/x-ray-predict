from SPRXRay import SPRXRay
from keras.applications import ResNet50
from keras.layers import Dense
from keras.models import Model


class XRayModelTrainer:
    def __init__(self):
        self.data = SPRXRay()
        self.base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        self.gender_model = None
        self.age_model = None

    def train_gender_model(self):
        # Add custom layers for gender classification
        gender_output = Dense(1, activation="sigmoid")(self.base_model.output)
        self.gender_model = Model(inputs=self.base_model.input, outputs=gender_output)

        # Freeze the base layers
        for layer in self.base_model.layers:
            layer.trainable = False

        (x_train, y_train), (_, _) = self.data.load_data(type_of_predict="gender")

        # Compile and train the gender classification model
        self.gender_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.gender_model.fit(x_train, y_train, epochs=10, batch_size=32)

        return self.gender_model

    def train_age_model(self):
        # Add custom layers for age regression
        age_output = Dense(1)(self.base_model.output)
        self.age_model = Model(inputs=self.base_model.input, outputs=age_output)

        # Freeze the base layers
        for layer in self.base_model.layers:
            layer.trainable = False

        (x_train, y_train), (_, _) = self.data.load_data(type_of_predict="age")

        # Compile and train the age regression model
        self.age_model.compile(optimizer="adam", loss="mean_squared_error")
        self.age_model.fit(x_train, y_train, epochs=10, batch_size=32)

        return self.age_model

    def train_models(self):
        return (self.train_gender_model(), self.train_age_model())
    
trainer = XRayModelTrainer()
gender_model, age_model = trainer.train_models()

