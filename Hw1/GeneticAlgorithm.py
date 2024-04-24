import numpy as np
import random
import math
import matplotlib.pyplot as plt
import numpy as np

class GeneticAlgorithm:
    # N：Initial population 的大小，此變數決定初始群集內 chromosome 的個數。
    # D：Dimension 維度大小，欲求解目標問題之變數數量。
    # B：Bits number，D 維度中每個變數可能出現的數字受限於 bit 數的控制。
    # n：於每次 Iteration 中從 population 抓取菁英個數的數量。
    # cr：Crossover rate，交配之門檻值。
    # mr：Mutation rate，突變之門檻值。
    # max_iter：欲進行迭代之總次數。
    # BD : x之上下邊界數值
    def __init__(self, Nnumber=10, 
                        # 第一題
                        Dimension=2,
                        # 第二題
                        # Dimension=2,
                        # 第三題
                        # Dimension=2, 
                        # 第四題
                        # Dimension=3, 
                        Bitnum=10, Elite_num=4, CrossoverRate=0.9, 
                       MutationRate=0.02, MaxIteration=1000, 
                    # 第一題
                       Boundary = [-3, 3] 
                    # 第二題
                    #    Boundary = [-1000, 1000] 
                    # 第三題
                    #    Boundary = [-5, 5] 
                    # 第四題
                    #    Boundary = [-5, 5] 
                       ):
        self.N = Nnumber
        self.D = Dimension
        self.B = Bitnum
        self.n = Elite_num
        self.cr = CrossoverRate
        self.mr = MutationRate
        self.max_iter = MaxIteration
        self.BD = Boundary
    # 生成Population
    def generatePopulation(self):
        population = []
        for number in range(self.N):
            chrom_list = []
            for run in range(self.D):
                element = (np.zeros((1,self.B))).astype(int)
                for i in range(1):
                    for j in range(self.B):
                        element[i,j] = np.random.randint(0,2)
                # print(39, element[0])
                chromosome = list(element[0])
                chrom_list.append(chromosome)
            population.append(chrom_list)
        return population
    
    #  二進制轉十進制 (Decoding)
    def B2D(self, pop):
        # print(64, pop)
        # print(65, len(pop))
        dec = (np.sum(x * (2**i)  for i, x in enumerate(pop)) / 2**len(pop)) * (self.BD[1] - self.BD[0]) + self.BD[0]
        # print(71, dec)
        return dec
    
    # 十進制轉二進制
    def D2B(self, num):
        return [int(i) for i in (bin(10)[2:])]
    
    # 第一題 Rosenbrock function
    def fun1(self, pop):
        X = np.array(pop)
        funsum = 0
        for i in range(self.D - 1):
            x1 = X[:,i]
            x2 = X[:,i+1]
            funsum += 100*(x2-x1**2)**2+(1 - x1)**2
        return list(funsum)
    

    # 第二題 Eggholder function
    def fun2(self, pop):
        X = np.array(pop)
        # print(102, X)
        funsum = 0
        for i in range(self.D - 1):
            x = X[:, i]
            # print(106, "x", x)
            y = X[:, i + 1]
            # print(107, "y", y)
            term1 = -(y + 47) * np.sin(np.sqrt(np.abs(y/2 + x + 47)))
            term2 = -(x * np.sin(np.sqrt(np.abs(x - (y + 47)))))
            funsum += term1 + term2
        # print(112, funsum)
        return list(funsum)
    
    
    # 第三題 Himmelblau's function
    def fun3(self, pop):
        X = np.array(pop)
        # print(119, X)
        funsum = 0
        for i in range(self.D - 1):
            x = X[:,i]
            # print(122, "x", x)
            y = X[:, i + 1]
            # print(124, "y", y)
            funsum += ((x**2+y-11)**2) + (((x+y**2-7)**2))
        return list(funsum)
    

    # 第四題 Rastrigin function
    def fun4(self, pop):
        X = np.array(pop)
        funsum = 0
        for i in range(self.D):
            x = X[:,i]
            funsum += x**2 - 10*np.cos(2*np.pi*x)
        funsum += 10*self.D
        return list(funsum)
    
    # 選擇方法: 俄羅斯輪盤
    def Selection(self, n, pop_bin, fitness):
        select_bin = pop_bin.copy()
        fitness1 = fitness.copy()
        Parents = []
        if sum(fitness1) == 0:
            for i in range(self.n):
                parent = select_bin[random.randint(0,(self.N)-1)]
                Parents.append(parent)
        else: 
            NorParent = [(1 - indivi/sum(fitness1))/((self.N-1)) for indivi in fitness1]
            tep = 0
            Cumulist = []
            for i in range(len(NorParent)):
                tep += NorParent[i]
                Cumulist.append(tep)
            #Find parents
            for i in range(self.n):
                z1 = random.uniform(0,1)
                for pick in range(len(Cumulist)):
                    if z1<=Cumulist[0]:
                        parent = select_bin[NorParent.index(NorParent[0])]
                    elif Cumulist[pick] < z1 <=Cumulist[pick+1]:
                        parent = select_bin[NorParent.index(NorParent[pick+1])]
                Parents.append(parent)
        return Parents
    
    # Crossover & Mutation
    def Crossover_Mutation(self, parent1, parent2):
        def swap_machine(element_1, element_2):
            temp = element_1
            element_1 = element_2
            element_2 = temp
            return element_1, element_2
        child_1 = []
        child_2 = []
        for i in range(len(parent1)):
            #隨機生成一數字，用以決定是否進行Crossover
            z1 = random.uniform(0,1)
            if z1 < self.cr:
                z2 = random.uniform(0,1)
                #決定要交換的位置點
                cross_location = math.ceil(z2*(len(parent1[i])-1))
                #Crossover
                parent1[i][:cross_location],parent2[i][:cross_location] = swap_machine(parent1[i][:cross_location],parent2[i][:cross_location])
                p_list = [parent1[i], parent2[i]]
                #隨機生成一數字，用以決定是否進行mutation
                for i in range(len(p_list)):
                    z3 = random.uniform(0,1)
                    if z3 < self.mr:
                        #決定要mutate的數字
                        z4 = random.uniform(0,1)
                        temp_location = z4*(len(p_list[i])-1)
                        mutation_location = 0 if temp_location < 0.5 else math.ceil(temp_location)
                        p_list[i][mutation_location] = 0 if p_list[i][mutation_location] == 1 else 1
                child_1.append(p_list[0])
                child_2.append(p_list[1])
            else:
                child_1.append(parent1[i])
                child_2.append(parent2[i])
        return child_1,child_2

def main():
    ga = GeneticAlgorithm()
    print(ga.N, ga.D, ga.B)
    pop_bin = ga.generatePopulation()
    print("all genetic: ", pop_bin)
    pop_dec = []
    for i in range(ga.N):
        chrom_rv = []
        for j in range(ga.D):
            chrom_rv.append(ga.B2D(pop_bin[i][j]))
        pop_dec.append(chrom_rv)

    print("binary decoding: ", pop_dec)
    # # 第一題 Rosenbrock function
    fitness = ga.fun1(pop_dec)
    # # 第二題 Eggholder function
    # fitness = ga.fun2(pop_dec)
    # # 第三題 Himmelblau's function
    # fitness = ga.fun3(pop_dec)
    # 第四題 Rastrigin function
    # fitness = ga.fun4(pop_dec)
    
    print("caculate fitness: ", fitness)

    best_rvlist = []
    best_valuelist = []
    
    it = 0
    while it < ga.max_iter:
        Parents_list = ga.Selection(ga.n, pop_bin, fitness)
        Offspring_list = []
        for i in range(int((ga.N-ga.n)/2)):
            candidate = [Parents_list[random.randint(0,len(Parents_list)-1)] for i in range(2)]
            # print(146, candidate)
            after_cr_mu = ga.Crossover_Mutation(candidate[0], candidate[1])
            offspring1, offspring2 = after_cr_mu[0], after_cr_mu[1]
            Offspring_list.append(offspring1)
            Offspring_list.append(offspring2)

        final_bin = Parents_list + Offspring_list
        print(164, final_bin)
        final_dec = []

        for i in range(ga.N):
            rv = []
            for j in range(ga.D):
                # print(169, ga.B2D(final_bin[i][j]))
                rv.append(ga.B2D(final_bin[i][j]))
            final_dec.append(rv)

        print(251, final_dec)

        # # Final fitness 第一題 Rosenbrock function
        final_fitness = ga.fun1(final_dec)
        # # Final fitness 第二題 Eggholder function
        # final_fitness = ga.fun2(final_dec)
        # # Final fitness 第三題 Himmelblau's function
        # final_fitness = ga.fun3(final_dec)
        # Final fitness 第四題 Rastrigin function
        # final_fitness = ga.fun4(final_dec)

        print("result fitness: ", final_fitness)
        #Take the best value in this iteration
        smallest_fitness = min(final_fitness)
        index = final_fitness.index(smallest_fitness)
        smallest_dec = final_dec[index]

        #Store the best fitness in the list
        best_rvlist.append(smallest_dec)
        best_valuelist.append(smallest_fitness)

        #Parameters back to the initial
        pop_bin = final_bin 
        pop_dec = final_dec
        fitness = final_fitness

        it += 1
    
    #Store best result
    every_best_value = []
    every_best_value.append(best_valuelist[0])
    for i in range(ga.max_iter-1):
        if every_best_value[i] >= best_valuelist[i+1]:
            every_best_value.append(best_valuelist[i+1])

        elif every_best_value[i] <= best_valuelist[i+1]:
            every_best_value.append(every_best_value[i])

    print('The best fitness: ', min(best_valuelist))
    best_index = best_valuelist.index(min(best_valuelist))
    print('Setup list is: ')
    print(best_rvlist[best_index])

    plt.figure(figsize = (15,8))
    plt.xlabel("Iteration",fontsize = 15)
    plt.ylabel("Fitness",fontsize = 15)

    plt.plot(every_best_value,linewidth = 2, label = "Best fitness convergence", color = 'b')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()