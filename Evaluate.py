import torch 
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class evaluate:
    def __init__(self, model, test_loader, threshold=0.4, DISEASES=None):
        self.model = model.eval()
        self.test_loader = test_loader
        self.threshold = threshold
        self.DISEASES = DISEASES 

    def _get_thresholds(self, num_classes):

        if isinstance(self.threshold, (float, int)):
            return np.array([float(self.threshold)] * num_classes, dtype=np.float32)

        if isinstance(self.threshold, (list, tuple, np.ndarray)):
            thresholds = np.array(self.threshold, dtype=np.float32)
            if len(thresholds) != num_classes:
                raise ValueError(
                    f"threshold 長度 {len(thresholds)} 與類別數 {num_classes} 不一致"
                )
            return thresholds

        raise TypeError("threshold 必須是 float/int 或 list/tuple/np.ndarray")


    def evaluate(self):
        device = next(self.model.parameters()).device

        test_preds = []
        test_trues = []

        # test 評估
        with torch.no_grad():
            for x, y, _ in self.test_loader:
                x = x.to(device)

                output = self.model(x)
                prob = torch.sigmoid(output).cpu()

                test_preds.append(prob)
                test_trues.append(y.cpu())

        # 合併 batch
        test_preds = torch.cat(test_preds, dim=0).numpy()   # 機率值 shape [N, C]
        test_trues = torch.cat(test_trues, dim=0).numpy()   # 真實標籤 shape [N, C]

        num_classes = test_preds.shape[1]
        thresholds = self._get_thresholds(num_classes)

        # AUC 用原始機率，不受 threshold 影響
        test_auc = roc_auc_score(
            test_trues,
            test_preds,
            average="macro"
        )

        # 每類別 threshold 二值化
        test_binary = (test_preds >= thresholds.reshape(1, -1)).astype(int)

        # macro 指標
        test_f1 = f1_score(
            test_trues,
            test_binary,
            average="macro",
            zero_division=0
        )

        test_recall = recall_score(
            test_trues,
            test_binary,
            average="macro",
            zero_division=0
        )

        test_precision = precision_score(
            test_trues,
            test_binary,
            average="macro",
            zero_division=0
        )

        print("===== Overall =====")
        print("Test AUC       :", round(test_auc, 4))
        print("Test F1-score  :", round(test_f1, 4))
        print("Test Recall    :", round(test_recall, 4))
        print("Test Precision :", round(test_precision, 4))

        # per-class 指標
        print("\n===== Per-class =====")
        for i, name in enumerate(self.DISEASES):
            y_true = test_trues[:, i]
            y_prob = test_preds[:, i]
            y_pred = test_binary[:, i]

            pos_count = int(y_true.sum())
            neg_count = int(len(y_true) - y_true.sum())

            if len(np.unique(y_true)) < 2:
                auc = None
            else:
                auc = roc_auc_score(y_true, y_prob)

            f1 = f1_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)

            print(f"\n{name}")
            print("  threshold :", float(thresholds[i]))
            print("  positives :", pos_count)
            print("  negatives :", neg_count)
            print("  AUC       :", None if auc is None else round(auc, 4))
            print("  F1        :", round(f1, 4))
            print("  Recall    :", round(rec, 4))
            print("  Precision :", round(prec, 4))