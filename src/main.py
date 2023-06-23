from SPRXRay import SPRXRay
from CNNFineTunning import XRayModelTrainer


from utils.read_image import read_image
from utils.image_show import image_show

import numpy as np

if __name__ == "__main__":
    # data = SPRXRay()
    trainer = XRayModelTrainer()

    gender_model, age_model = trainer.train_models()

    img_path = (
        "/home/vinicius/repositories/x-ray-predict/data/kaggle/kaggle/train/008386.png"
    )

    image_show(img_path)
    img = read_image(img_path)

    img = np.expand_dims(img, axis=0)

    img = img / 255.0

    print(img)

    prediction = age_model.predict(img)
    print(int(prediction[0][0]))

    prediction = gender_model.predict(img)
    if prediction[0] < 0.5:
        print("Male")
    else:
        print("Female")
