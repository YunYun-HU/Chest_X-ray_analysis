import random
from CNN import CNNModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
#chromosome
BASE_FILTERS_CHOICES = [16, 32, 64]
DENSE_UNITS_CHOICES  = [128, 256, 512]
DROPOUT_CHOICES      = [0.2, 0.3, 0.4, 0.5]
LR_CHOICES           = [1e-4, 3e-4, 1e-3]
BATCH_SIZE_CHOICES   = [8, 16, 32]

class GeneticOptimizer:
    #代數，族群數，精英數，突變率
    def __init__(self, input_size, num_classes, picture_train, lab_train, picture_val, lab_val,
                 pop_size=8, generations=5, elite_size=2, mutation_rate=0.2):
        self.input_size = input_size
        self.num_classes = num_classes
        self.picture_train = picture_train
        self.lab_train = lab_train
        self.picture_val = picture_val
        self.lab_val = lab_val

        self.pop_size = pop_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    #隨機產生一條染色體
    def random_chromosome(self):
        return [
            random.randint(0, len(BASE_FILTERS_CHOICES)-1),
            random.randint(0, len(DENSE_UNITS_CHOICES)-1),
            random.randint(0, len(DROPOUT_CHOICES)-1),
            random.randint(0, len(LR_CHOICES)-1),
            random.randint(0, len(BATCH_SIZE_CHOICES)-1),
        ]

    #染色體解碼成參數 like "base_filters": 32,
    def decode(self, chromosome):
        return {
            "base_filters": BASE_FILTERS_CHOICES[chromosome[0]],
            "dense_units": DENSE_UNITS_CHOICES[chromosome[1]],
            "dropout_rate": DROPOUT_CHOICES[chromosome[2]],
            "lr": LR_CHOICES[chromosome[3]],
            "batch_size": BATCH_SIZE_CHOICES[chromosome[4]],
        }

    #計算染色體分數
    def fitness(self, chromosome):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = self.decode(chromosome)

        #呼叫cnn
        model = CNNModel(
        self.input_size,
        self.num_classes,
        base_filters=params["base_filters"],
        dense_units=params["dense_units"],
        dropout_rate=params["dropout_rate"]
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        train_ds = TensorDataset(self.picture_train, self.lab_train)
        val_ds   = TensorDataset(self.picture_val, self.lab_val)

        train_loader = DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=params["batch_size"],
            shuffle=False
        )

        # train
        model.train()
        for epoch in range(2):   # GA 階段先少跑
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # validation
        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)

                output = model(x)
                prob = torch.sigmoid(output).cpu()

                preds.append(prob)
                trues.append(y)

        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()

        score = roc_auc_score(trues, preds, average="macro")

        return score

    #選擇父母
    def select_parents(self, population, scores):

        # tournament selection
        selected = []
        for _ in range(2):
            idxs = random.sample(range(len(population)), 3)
            best_idx = max(idxs, key=lambda i: scores[i])
            selected.append(population[best_idx])
        return selected

    #交配
    def crossover(self, p1, p2):
        point = random.randint(1, len(p1)-1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    #突變
    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                if i == 0:
                    chromosome[i] = random.randint(0, len(BASE_FILTERS_CHOICES)-1)
                elif i == 1:
                    chromosome[i] = random.randint(0, len(DENSE_UNITS_CHOICES)-1)
                elif i == 2:
                    chromosome[i] = random.randint(0, len(DROPOUT_CHOICES)-1)
                elif i == 3:
                    chromosome[i] = random.randint(0, len(LR_CHOICES)-1)
                elif i == 4:
                    chromosome[i] = random.randint(0, len(BATCH_SIZE_CHOICES)-1)
        return chromosome


    #執行
    def run(self):
        #初始化族群
        population = [self.random_chromosome() for _ in range(self.pop_size)]

        #初始化最佳染色體
        best_chromosome = None
        best_score = -1

        #迭代
        for gen in range(self.generations):
            #計算分數
            scores = [self.fitness(ch) for ch in population]

            #依分數排序
            ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            population = [x[0] for x in ranked]
            scores = [x[1] for x in ranked]

            #更新最佳染色體
            if scores[0] > best_score:
                best_score = scores[0]
                best_chromosome = population[0]

            print(f"Generation {gen+1}: best_val_auc = {scores[0]:.4f}, best = {self.decode(population[0])}")
            
            #保留菁英
            new_population = population[:self.elite_size]

            #產生新族群 
            while len(new_population) < self.pop_size:
                p1, p2 = self.select_parents(population, scores)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            #用新族群進入下一代
            population = new_population

        return best_chromosome, best_score, self.decode(best_chromosome)