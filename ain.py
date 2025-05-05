class ArtificialImmuneNetwork:
    def __init__(self, fitness_func, num_parameters, population_size, num_generations, clone_factor, mutation_rate, suppression_threshold, early_stopping_patience):
        self.fitness_func = fitness_func
        self.num_parameters = num_parameters
        self.population_size = population_size
        self.num_generations = num_generations
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.suppression_threshold = suppression_threshold
        self.early_stopping_patience = early_stopping_patience
        

    def initialize_population(self):
        # 使用拉丁超立方抽样生成初始种群
        initial_population = lhs(self.num_parameters, samples=self.population_size)
        return initial_population

    def clone(self, population, fitness_values):
        # 计算最大适应度值
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        # 计算个体的调整适应度值,确保值在(0, 1]范围内
        adjusted_fitness = (max_fitness - fitness_values + 1e-10) / (max_fitness - min_fitness + 1e-10)

        clones = []
        for antibody, fitness in zip(population, adjusted_fitness):
            # 计算克隆数量,根据调整适应度值和克隆因子
            num_clones = int(self.clone_factor * fitness)

            # 生成克隆个体
            clones.extend([antibody.copy() for _ in range(num_clones)])

        return np.array(clones)

    def mutate(self, clones,fitness_values):
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        adjusted_fitness = (max_fitness - fitness_values + 1e-10) / (max_fitness - min_fitness + 1e-10)
        mutation_rates = self.mutation_rate * (1 - adjusted_fitness)
        for i in range(len(clones)):
            mutation_mask = np.random.rand(self.num_parameters) < mutation_rates[i]
            gaussian_noise = np.random.normal(0, self.mutation_rate, size=self.num_parameters)  # 匹配每个克隆的维度
            clones[i][mutation_mask] += gaussian_noise[mutation_mask]  # 应用变异掩码
            clones[i] = np.clip(clones[i], 0, 1)  # 确保值在0和1之间
        return clones
    
    def suppress(self, antibodies):
        suppressed_antibodies = []
        for i in range(len(antibodies)):
            too_close = False
            for j in range(len(suppressed_antibodies)):
                if np.linalg.norm(antibodies[i] - suppressed_antibodies[j]) < self.suppression_threshold:
                    too_close = True
                    break
            if not too_close:
                suppressed_antibodies.append(antibodies[i])
        return np.array(suppressed_antibodies)
    
    # def suppress(self, antibodies):
    #     if len(antibodies) == 0:
    #         return np.array([])

    #     kdt = KDTree(antibodies, leaf_size=30, metric='euclidean')
    #     suppressed_antibodies = []
    #     indices_to_remove = set()

    #     for i in range(len(antibodies)):
    #         if i in indices_to_remove:
    #             continue
    #         suppressed_antibodies.append(antibodies[i])
    #         ind = kdt.query_radius(antibodies[i].reshape(1, -1), r=self.suppression_threshold)[0]
    #         indices_to_remove.update(ind.tolist())

    #     return np.array(suppressed_antibodies)

    # def network_interaction(self, population, fitness_values):
    #     # 计算个体之间的距离矩阵
    #     distances = np.zeros((len(population), len(population)))
    #     for i in range(len(population)):
    #         for j in range(i + 1, len(population)):
    #             distances[i, j] = distances[j, i] = np.linalg.norm(population[i] - population[j])

    #     # 计算个体之间的亲和力
    #     affinities = 1 / (distances + 1e-10)  # 添加一个小的正数,以避免除以零

    #     # 根据亲和力对个体进行交互
    #     for i in range(len(population)):
    #         for j in range(len(population)):
    #             if i != j:
    #                 interaction_strength = affinities[i, j] * (fitness_values[j] - fitness_values[i])
    #                 population[i] += interaction_strength * (population[j] - population[i])

    #     return population


    def optimize(self):
        population = self.initialize_population()
        best_fitness = np.inf
        patience_counter = 0

        for generation in range(self.num_generations):
            fitness_values = np.apply_along_axis(self.fitness_func, 1, population)
            clones = self.clone(population, fitness_values)
            mutated_clones = self.mutate(clones, fitness_values)
            combined_population = np.vstack((population, mutated_clones))
            combined_population = self.suppress(combined_population)

            # 检查个体数量是否达到要求，进行填补或调整
            num_current_population = combined_population.shape[0]
            if num_current_population < self.population_size:
                additional_population = self.initialize_population()[:self.population_size - num_current_population]
                combined_population = np.vstack((combined_population, additional_population))

            combined_fitness = np.apply_along_axis(self.fitness_func, 1, combined_population)
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]

            current_best_fitness = np.min(combined_fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at generation {generation}")
                    break

        best_index = np.argmin(np.apply_along_axis(self.fitness_func, 1, population))
        return population[best_index]