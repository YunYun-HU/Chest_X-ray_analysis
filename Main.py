from OpenFile import Readfile
from Data_Cleaning import DataCleaning
from GA import GeneticOptimizer
from DenseNet import DenseNet
from Evaluate import evaluate

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

    #保留疾病
    DISEASES = [
        "Infiltration",
        "Effusion"
        ]

    #data cleaning
    cleaner = DataCleaning(csv_path, png_path, input_size=224, DISEASES=DISEASES)
    df  = cleaner.clean_cxr8_csv()
    print(df.head())
    #切分
    picture_train, picture_val, picture_test, lab_train, lab_val, lab_test = cleaner.split_data(df)

    #ga
    ga = GeneticOptimizer(
        input_size=224,
        num_classes=len(cleaner.DISEASES),
        picture_train=picture_train,
        lab_train=lab_train,
        picture_val=picture_val,
        lab_val=lab_val,
        pop_size=8,
        generations=1,
        elite_size=2,
        mutation_rate=0.2
    )

    best_chromosome, best_score, best_params = ga.run()

    print("Best chromosome:", best_chromosome)
    print("Best val_auc:", best_score)
    print("Best params:", best_params)


    #用最佳參數建立最終模型
    model = DenseNet(
        input_size=224,
        num_classes=len(cleaner.DISEASES),
        dense_units=best_params["dense_units"],
        dropout_rate=best_params["dropout_rate"]
    ).to(device)

    pos_counts = lab_train.sum(dim=0)
    neg_counts = len(lab_train) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print("pos_weight =", pos_weight)

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

    
    #評估資料
    evaluator = evaluate(
    model=model,
    test_loader=test_loader,
    threshold=[0.1, 0.1],
    DISEASES=DISEASES
    )

    evaluator.evaluate()

if __name__ == "__main__":
    main()