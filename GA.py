import numpy as np
import torch

from sklearn.metrics import f1_score, recall_score, precision_score


class GAThresholdOptimizer:
    def __init__(
        self,
        model,
        val_loader,
        device,
        num_classes,
        population_size=30,
        generations=50,
        elite_size=2,
        mutation_rate=0.2,
        crossover_rate=0.8,
        threshold_min=0.05,
        threshold_max=0.95,
        fitness_mode="f1"
    ):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes

        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

        self.fitness_mode = fitness_mode


    # 收集 validation set 的預測機率與真實標籤
    def collect_val_outputs(self):
        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:

                # 你的 main 裡面 dataset 回傳的是 x, y, sample_weight
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch

                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                prob = torch.sigmoid(output)

                all_probs.append(prob.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        return all_probs, all_labels


    # 初始化族群
    # 每一條 chromosome 都是一組 threshold
    def initialize_population(self):
        population = np.random.uniform(
            low=self.threshold_min,
            high=self.threshold_max,
            size=(self.population_size, self.num_classes)
        )

        return population


    # fitness function
    def fitness(self, chromosome, y_true, y_prob):
        y_pred = (y_prob >= chromosome).astype(int)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        if self.fitness_mode == "f1":
            score = macro_f1

        elif self.fitness_mode == "medical":
            # 醫療影像通常比較重視不要漏診
            # 所以 recall 權重稍微提高
            score = (
                0.5 * macro_f1 +
                0.3 * macro_recall +
                0.2 * macro_precision
            )

        elif self.fitness_mode == "recall":
            score = macro_recall

        else:
            score = macro_f1

        return score


    # selection：錦標賽選擇
    def selection(self, population, fitness_scores):
        tournament_size = 3

        selected_indices = np.random.choice(
            len(population),
            size=tournament_size,
            replace=False
        )

        selected_fitness = fitness_scores[selected_indices]
        best_index = selected_indices[np.argmax(selected_fitness)]

        return population[best_index].copy()


    # crossover：均勻交配
    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        mask = np.random.rand(self.num_classes) < 0.5

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]

        return child1, child2


    # mutation：突變 threshold
    def mutate(self, chromosome):
        for i in range(self.num_classes):
            if np.random.rand() < self.mutation_rate:
                noise = np.random.normal(
                    loc=0.0,
                    scale=0.05
                )

                chromosome[i] += noise

                chromosome[i] = np.clip(
                    chromosome[i],
                    self.threshold_min,
                    self.threshold_max
                )

        return chromosome


    # GA 主流程
    def optimize(self):
        y_prob, y_true = self.collect_val_outputs()

        population = self.initialize_population()

        best_chromosome = None
        best_score = -1

        for generation in range(self.generations):
            fitness_scores = np.array([
                self.fitness(chromosome, y_true, y_prob)
                for chromosome in population
            ])

            sorted_indices = np.argsort(fitness_scores)[::-1]

            current_best_index = sorted_indices[0]
            current_best_score = fitness_scores[current_best_index]
            current_best_chromosome = population[current_best_index]

            if current_best_score > best_score:
                best_score = current_best_score
                best_chromosome = current_best_chromosome.copy()

            print(
                f"[GA Threshold] Generation {generation + 1}/{self.generations} | "
                f"Best Fitness = {best_score:.4f} | "
                f"Threshold = {np.round(best_chromosome, 3)}"
            )

            new_population = []

            # elite 保留
            elites = population[sorted_indices[:self.elite_size]]

            for elite in elites:
                new_population.append(elite.copy())

            # 產生下一代
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population)

        return best_chromosome.tolist(), best_score