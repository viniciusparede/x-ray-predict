from sklearn.model_selection import train_test_split
from SPRXRay import SPRXRay


if __name__ == "__main__":
    data = SPRXRay()
    (X_train, y_train), (X_test, y_test) = data.load_data(type_of_predict="age")
    


    print(X_train.shape)