from OpenFile import Readfile
from Data_Cleaning import DataCleaning
from CNN import CNNModel
from GA import GeneticOptimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def main():
    #openfile
    loader = Readfile()
    csv_path = loader.read_csv()
    png_path = loader.read_png()

    #data cleaning
    cleaner = DataCleaning(csv_path, png_path, input_size=224)
    df  = cleaner.clean_cxr8_csv()

    #切分
    picture_train, picture_val, picture_test, lab_train, lab_val, lab_test = cleaner.split_data(df)

    print(df[cleaner.DISEASES].sum())    
    
    #ga
    ga = GeneticOptimizer(
        input_size=224,
        num_classes=len(cleaner.DISEASES),
        picture_train=picture_train,
        lab_train=lab_train,
        picture_val=picture_val,
        lab_val=lab_val,
        pop_size=8,
        generations=4,
        elite_size=2,
        mutation_rate=0.2
    )

    best_chromosome, best_score, best_params = ga.run()

    print("Best chromosome:", best_chromosome)
    print("Best val_auc:", best_score)
    print("Best params:", best_params)


    #用最佳參數建立最終模型
    model = CNNModel(
        input_size=224,
        num_classes=len(cleaner.DISEASES),
        base_filters=best_params["base_filters"],
        dense_units=best_params["dense_units"],
        dropout_rate=best_params["dropout_rate"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])


    #DataLoader
    train_loader = DataLoader(
        TensorDataset(picture_train, lab_train),
        batch_size=best_params["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(picture_val, lab_val),
        batch_size=best_params["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(picture_test, lab_test),
        batch_size=best_params["batch_size"],
        shuffle=False
    )


    #訓練最終模型
    epochs = 15

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        #驗證
        model.eval()
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                output = model(x)
                prob = torch.sigmoid(output).cpu()

                val_preds.append(prob)
                val_trues.append(y)

        val_preds = torch.cat(val_preds).numpy()
        val_trues = torch.cat(val_trues).numpy()
        val_auc = roc_auc_score(val_trues, val_preds, average="macro")

        print(f"Epoch {epoch+1}/{epochs}, val_auc={val_auc:.4f}")


    #test評估
    model.eval()
    test_preds = []
    test_trues = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            prob = torch.sigmoid(output).cpu()

            test_preds.append(prob)
            test_trues.append(y)

    test_preds = torch.cat(test_preds).numpy()
    test_trues = torch.cat(test_trues).numpy()
    test_auc = roc_auc_score(test_trues, test_preds, average="macro")

    print("Test AUC:", test_auc)


if __name__ == "__main__":
    main()