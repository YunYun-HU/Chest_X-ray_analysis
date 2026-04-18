from OpenFile import Readfile
from Data_Cleaning import DataCleaning
from CNN import CNNModel
def main():
    #openfile
    loader = Readfile()
    csv_path = loader.read_csv()
    png_path = loader.read_png()

    #data cleaning
    cleaner = DataCleaning(csv_path, png_path)
    clean_csv = cleaner.clean_cxr8_csv()

    #切分
    X_train, X_val, X_test, y_train, y_val, y_test = cleaner.split_data(clean_csv)

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)
    

    #cnn
    #cnn_model = CNNModel(X_train, X_test)
    #cnn_model.build_model()


if __name__ == "__main__":
    main()