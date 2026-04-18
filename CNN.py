from OpenFile import Readfile
from keras import models, layers, optimizers, metrics, Input

class CNNModel:
    def __init__(self, csv_data, png_data):
        #我給當inputfile
        self.reader = Readfile()
        self.csv_data = csv_data
        self.png_data = png_data



    def build_model(
        input_size=512,
        conv1=32,
        conv2=64,
        conv3=128,
        conv4=256,
        conv5=256,
        dense_units=256,
        dropout_rate=0.4,
        lr=1e-3,
        num_classes=14,
        ):

        model = models.Sequential([
            Input(shape=(input_size, input_size, 1)),

            layers.Conv2D(conv1, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.Conv2D(conv2, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.Conv2D(conv3, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.Conv2D(conv4, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.Conv2D(conv5, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='sigmoid')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=[
                metrics.BinaryAccuracy(name='bin_acc'),
                metrics.AUC(name='auc', multi_label=True),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall')
            ]
        )
        return model