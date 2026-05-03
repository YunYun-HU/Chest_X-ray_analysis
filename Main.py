from OpenFile import OpenFile
from Data_Process import DataCleaning, DataForDenseNet
from GA import GAThresholdOptimizer
from DenseNet import DenseNet, EarlyStopping
from Evaluate import evaluate
from GradCAM import GradCAM, show_gradcam

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

def main():
    #--------------------------------基本設定---------------------------------
    
    #第幾個訓練集
    number = 3

    #--------------------------------超參數---------------------------------
    #輸入大小
    input_size = 340

    #gpu顯存用量
    batch_size = 32

    epochs = 20
    lr = 1e-4

    #中間層數量
    dense_units = 256
    dropout_rate = 0.3

    #CPU加速讀資料
    num_workers = 6

    #超過多少判斷有病
    threshold = [0.4, 0.35, 0.81, 0.74]                      #[0.5, 0.5, 0.7, 0.5, 0.7, 0.55, 0.7]    Effusion Atelectasis Cardiomegaly Pneumothorax  Edema
                                                                        # Nodule Emphysema 
    
    #使否使用ga
    use_ga = True
    
    #是否使用elvaluate評估
    use_evaluate = True
    
    #使用已經訓練好的 DenseNet 權重。
    pretrained = True

    #早停
    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.001
    )

    #-----------------------------資料集疾病標籤---------------------------------
 
    #保留疾病
    DISEASES = [
        "Effusion",
        "Atelectasis",
        "Cardiomegaly",
        "Emphysema"
        ]
    
    
    num_classes = len(DISEASES)

    #-----------------------------路徑設定---------------------------------

    #判斷路徑
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent  
    csv_file = PROJECT_DIR / "CXR8" / f"set{number}" / f"source_{num_classes}feature.csv"
    CXR8_DIR = PROJECT_DIR / "CXR8"

    #--------------------------------gpu加速---------------------------------

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)


    #-----------------------------start 不完整要完全刪掉整個set---------------------------------

    #全域變數
    global train_set, val_set, test_set

    loader = OpenFile(number)
    if csv_file.exists():
        
        print("檔案已存在，直接讀取")
        train_set = loader.read_train_df()
        val_set   = loader.read_val_df()    
        test_set  = loader.read_test_df()

    #沒有檔案建檔
    else:
        print("source_檔案不存在，建立新檔")

        #openfile
        csv_path = loader.read_csv()
        png_path = loader.read_png()

        #data cleaning
        cleaner = DataCleaning(csv_path, png_path, input_size=224, DISEASES=DISEASES)
        df = cleaner.clean_cxr8_csv()

        df = loader.new_file(df, len(DISEASES))

        if False:
            #看大小跟特徵量
            cleaner.findfrature(df)
            print(len(png_path))

        #切分
        train_df, val_df, test_df = cleaner.split_data(df)

        loader.split_train_df(train_df)
        loader.split_val_df(val_df)
        loader.split_test_df(test_df)

        #讀檔
        train_set = loader.read_train_df()
        val_set   = loader.read_val_df()    
        test_set  = loader.read_test_df()


    #cnn
    #csv轉換pytorch格式
    train_dataset = DataForDenseNet(
        csv_path=train_set,
        data_root=CXR8_DIR,
        diseases=DISEASES,
        input_size=input_size,
        mode="train"
    )

    val_dataset = DataForDenseNet(
        csv_path=val_set,
        data_root=CXR8_DIR,
        diseases=DISEASES,
        input_size=input_size,
        mode="eval"
    )

    test_dataset = DataForDenseNet(
        csv_path=test_set,
        data_root=CXR8_DIR,
        diseases=DISEASES,
        input_size=input_size,
        mode="eval"
    )

    #DataLoader 是pytorch資料批次讀取器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )


    #DenseNet
    model = DenseNet(
        input_size=input_size,
        num_classes=num_classes,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    ).to(device)


    #加入個別樣本權重的損失函數
    train_labels = torch.stack([
        train_dataset[i][1] for i in range(len(train_dataset))
    ]).float()

    labels = train_labels.float()   # shape: [N, 5]

    pos_count = labels.sum(dim=0)
    neg_count = labels.shape[0] - pos_count

    pos_weight = neg_count / (pos_count + 1e-6)
    pos_weight = torch.clamp(pos_weight, max=3.0)

    criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(device),
    reduction="none"
    )


    #criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)       

    #訓練
    best_val_loss = float("inf")
    best_model_path = "best_model.pth"

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y, sample_weight in train_loader:
            x = x.to(device)  # x shape: [batch, 1, 224, 224]
            y = y.to(device)  # y shape: [batch, 3]
            sample_weight = sample_weight.to(device) # shape: [batch]

            optimizer.zero_grad()

            output = model(x)  # output shape: [batch, 3]


            loss = criterion(output, y)
            loss = loss * sample_weight.view(-1, 1)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)


        #驗證
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, sample_weight in val_loader:
                x = x.to(device)
                y = y.to(device)
                sample_weight = sample_weight.to(device)

                output = model(x)

                loss = criterion(output, y)
                loss = loss * sample_weight.view(-1, 1)
                loss = loss.mean()
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"train_loss={avg_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )

        #save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved. val_loss={best_val_loss:.4f}")


        #early stopping
        early_stopping(avg_val_loss)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 載回 validation 最好的模型
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model. best_val_loss={best_val_loss:.4f}")

    #ga調整threshold
    if use_ga:
        ga_threshold = GAThresholdOptimizer(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        population_size=40,
        generations=80,
        elite_size=4,
        mutation_rate=0.15,
        crossover_rate=0.8,
        threshold_min=0.35,
        threshold_max=0.9,
        fitness_mode="f1"
        )

        threshold, best_fitness = ga_threshold.optimize()

        print("GA best threshold:", threshold)
        print("GA best validation fitness:", best_fitness)



    #測試評估資料
    evaluator = evaluate(
    model=model,
    test_loader=test_loader,
    threshold=threshold,
    DISEASES=DISEASES
    )

    evaluator.evaluate()


    if use_evaluate:
        
        print("grad-CAM輸入格式如下:什麼病徵判斷(number) (空格) test.csv圖片編號。")
        while True:

            #指定一種已學會的疾病判斷 在test.csv中指定一張圖片從零開始
            class_idx, idx = input().split()  # 0=Cardiomegaly, 1=Effusion ...

            try:
                class_idx = int(class_idx)
                idx = int(idx)

                # 建立 Grad-CAM
                target_layer = model.backbone.features.denseblock4
                gradcam = GradCAM(model, target_layer)


                image, label, _ = test_dataset[idx]

                print("idx:", idx)
                print("Disease:", DISEASES[class_idx])
                print("True label:", label)

                # 丟進模型
                x = image.unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    output = model(x)
                    prob = torch.sigmoid(output)

                print("Pred prob:", prob.cpu().numpy())

                # 產生 Grad-CAM
                cam = gradcam.generate(x, class_idx=class_idx)

                # 顯示
                show_gradcam(
                    image_tensor=image,
                    cam=cam,
                    title=f"Grad-CAM: {DISEASES[class_idx]}"
                )

            except Exception as e:
                print("錯誤是：", type(e).__name__, e)


if __name__ == "__main__":
    main()