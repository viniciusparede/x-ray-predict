from git import Repo
import os
import pandas as pd

from utils.get_images import get_images
from utils.read_image import read_image


import numpy as np


class SPRXRay:
    def __init__(self) -> None:
        repo = Repo(os.getcwd())
        base_dir = repo.working_tree_dir

        self._data_dir = os.path.join(base_dir, "data")
        self._kaggle_dir = os.path.join(self._data_dir, "kaggle")
        self._kaggle_dir = os.path.join(self._kaggle_dir, "kaggle")
        self._train_dir = os.path.join(self._kaggle_dir, "train")
        self._test_dir = os.path.join(self._kaggle_dir, "test")

        self._train_age = pd.read_csv(
            filepath_or_buffer=os.path.join(self._data_dir, "train_age.csv"),
            usecols=["imageId", "age"],
        )
        self._train_gender = pd.read_csv(
            filepath_or_buffer=os.path.join(self._data_dir, "train_gender.csv"),
            usecols=["imageId", "gender"],
        )

        self._sample_submission_age = pd.read_csv(
            filepath_or_buffer=os.path.join(
                self._data_dir, "sample_submission_age.csv"
            ),
            usecols=["imageId", "age"],
        )
        self._sample_submission_gender = pd.read_csv(
            filepath_or_buffer=os.path.join(
                self._data_dir, "sample_submission_gender.csv"
            ),
            usecols=["imageId", "gender"],
        )

    def load_data(self, type_of_predict: str):
        train_images: list = get_images(directory=self._train_dir)[0:100]
        test_images: list = get_images(directory=self._test_dir)[0:100]

        x_train = np.array([read_image(img_path) for img_path in train_images])
        x_test = np.array([read_image(img_path) for img_path in test_images])

        #from utils.image_show import image_show
        #image_show(train_images[1])

        match type_of_predict:
            case "age":
                y_train = self._train_age["age"].values[0:100]
                y_train = np.expand_dims(y_train, axis=1)
                y_test = self._sample_submission_age["age"].values[0:100]
                y_test = np.expand_dims(y_test, axis=1)

            case "gender":
                y_train = self._train_gender["gender"].values[0:100]
                y_train = np.expand_dims(y_train, axis=1)
                y_test = self._sample_submission_gender["gender"].values[0:100]
                y_test = np.expand_dims(y_test, axis=1)

        return ((x_train, y_train), (x_test, y_test))
