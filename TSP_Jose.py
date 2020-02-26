"""
Author: Jose Angel Molina
date: 21/10/2018
file: TSP_Jose.py
"""
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import project1.TSP_Project.Individual as tsp

def test_automatization():

    # Declaration of constant variables
    n_iterations = 300
    n_individuals = 500
    mutation_rate = 0.2

    t1 = time.time()

    file = open('soluts.txt', 'w')

    # Battery of tests to launch
    set_of_instances = ['inst-0.tsp', 'inst-13.tsp', 'inst-16.tsp']
    set_of_operators = [[BasicTSP.uniform_crossover, BasicTSP.reciprocal_exchange_mutation, BasicTSP.random_selection],
                        [BasicTSP.cycle_crossover, BasicTSP.scramble_mutation, BasicTSP.random_selection],
                        [BasicTSP.uniform_crossover, BasicTSP.reciprocal_exchange_mutation, BasicTSP.roulette_wheel],
                        [BasicTSP.cycle_crossover, BasicTSP.reciprocal_exchange_mutation, BasicTSP.roulette_wheel],
                        [BasicTSP.cycle_crossover, BasicTSP.scramble_mutation, BasicTSP.roulette_wheel],
                        [BasicTSP.uniform_crossover, BasicTSP.scramble_mutation, BasicTSP.best_second]]

    # Execution of tests
    n = 0

    it = 0

    matrix_times = np.zeros((len(set_of_operators) * len(set_of_instances), n_iterations))
    matrix_fitness = np.zeros((len(set_of_operators) * len(set_of_instances), n_iterations))

    for i in range(len(set_of_operators)):

        for instance in set_of_instances:
            mean_tim = []
            best = float('inf')
            media = 0

            for tim in range(3):
                file.write('Test {0}, configuration {1}, instance {2} repetition {3}: \n'.format(n, i, instance, tim))
                n += 1
                ga = BasicTSP(instance, n_individuals, mutation_rate, n_iterations, set_of_operators[i])
                file, fitness_best, m_times, m_fitness = ga.search(file, matrix_times, matrix_fitness, it)
                media += fitness_best
                mean_tim.append(fitness_best)
                if fitness_best < best:
                    best = fitness_best

                file.write('\n\n\n')

            sorted(mean_tim)
            media /= 3
            file.write('Median: {0}, media: {1}, best of 3: {2}.\n'.format(mean_tim[1], media, best))
            it += 1

    t2 = time.time()

    total_time = (t2 - t1)
    file.write('\n Total execution time: {0}s'.format(round(total_time, 2)))

    time_inst_0 = plt
    time_inst_1 = plt

    # Plot instances of time and fitness
    for plot in range(3):

        time_inst_0.plot(matrix_times[plot], label='Conf1')
        time_inst_0.plot(matrix_times[plot + 3], label='Conf2')
        time_inst_0.plot(matrix_times[plot + 6], label='Conf3')
        time_inst_0.plot(matrix_times[plot + 9], label='Conf4')
        time_inst_0.plot(matrix_times[plot + 12], label='Conf5')
        time_inst_0.plot(matrix_times[plot + 15], label='Conf6')

        time_inst_0.xlabel('Iterations')
        time_inst_0.ylabel('Time cost')
        time_inst_0.title('Time cost over iterations. Instance {0}'.format(plot))

        time_inst_0.show()
        time_inst_0 = plt

        time_inst_1.plot(matrix_fitness[plot], label=['Conf1'])
        time_inst_1.plot(matrix_fitness[plot + 3], label=['Conf2'])
        time_inst_1.plot(matrix_fitness[plot + 6], label=['Conf3'])
        time_inst_1.plot(matrix_fitness[plot + 9], label=['Conf4'])
        time_inst_1.plot(matrix_fitness[plot + 12], label=['Conf5'])
        time_inst_1.plot(matrix_fitness[plot + 15], label=['Conf6'])

        time_inst_1.xlabel('Iterations')
        time_inst_1.ylabel('Fitness')
        time_inst_1.title('Average fitness over iterations. Instance {0}'.format(plot))

        time_inst_1.show()
        time_inst_1 = plt

class BasicTSP:

    '''
    BASICS INITIALIZATION
    '''

    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, ops):
        """
        Parameters and general variables
        """
        self.operators = ops
        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.read_instance()
        self.init_population()

    # Reads the given instance
    def read_instance(self):
        file = open('dataset/' + self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    # Saves the given
    def saveSolution(self,f_name, solution, cost):
        file = open(f_name, 'w')
        file.write(str(cost) + "\n")
        for city in solution:
            file.write(str(city) + "\n")
        file.close()

    # Initializes at the very beginning the population.
    def init_population(self):

        for i in range(0, self.popSize):
            individual = tsp.Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()

        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print("Best initial sol: ", self.best.getFitness())

    '''
    SELECTION
    '''
    def best_second(self):
        return self.population[0], self.population[1]

    # Selects 2 random individuals.
    def random_selection(self):

        indA = self.matingPool[random.randint(0, self.popSize-1)]
        indB = self.matingPool[random.randint(0, self.popSize-1)]
        return [indA, indB]

    # Using the roulette techniques, obtains 2 individuals according to its fitness.
    def roulette_wheel(self):

        ind1, ind2 = None, None

        total_fitness = sum([c.getFitness() for c in self.matingPool])

        random_value = random.uniform(0, 1)

        previous_probability = 0
        rank = 0
        for i in range(len(self.matingPool)):

            previous_probability += (self.matingPool[i].getFitness() / total_fitness)

            if random_value < previous_probability:
                rank = i
                ind1 = self.matingPool[i]
                break

        total_fitness = total_fitness - ind1.getFitness()
        random_value = random.uniform(0, 1)

        previous_probability = 0

        for i in range(len(self.matingPool)):

            if rank != i:

                previous_probability += (self.matingPool[i].getFitness() / total_fitness)

                if random_value < previous_probability:
                    ind2 = self.matingPool[i]
                    break

        return ind1, ind2

    '''
    CROSSOVER METHODS
    '''
    def uniform_crossover(self, indA, indB):

        # Default crossover uniform rate
        ratio = 0.5

        # Obtain gene chains from both parents
        genes_indA = indA.genes
        genes_indB = indB.genes

        # Get the number of genes non mutable.
        size = indA.genSize
        aleat = size * ratio

        # Select randomly what genes are not gonna be moved
        lista_ale = set(random.sample(range(size), round(aleat)))

        # Create offspring
        son1 = tsp.Individual(self.genSize, self.data)
        son_genoma_1 = [0] * self.genSize

        son2 = tsp.Individual(self.genSize, self.data)
        son_genoma_2 = [0] * self.genSize

        # Copy the immutable genes to the new gene chain
        set_a = set()

        for e in lista_ale:
            son_genoma_1[e] = genes_indA[e]
            set_a.add(genes_indA[e])

        set_b = set()

        for e in lista_ale:
            son_genoma_2[e] = genes_indB[e]
            set_b.add(genes_indB[e])

        itA = 0
        itB = 0

        # Move genes left, from parentA to offspringB and from parentB to offspringA
        for i in range(self.genSize):

            if son_genoma_1[i] == 0:

                for h in range(itA, self.genSize + 1):
                    if not genes_indB[h] in set_a:
                        son_genoma_1[i] = genes_indB[h]
                        itA += 1
                        break

                    itA += 1


                for h in range(itB, self.genSize + 1):
                    if not genes_indA[h] in set_b:
                        son_genoma_2[i] = genes_indA[h]
                        itB += 1
                        break
                    itB += 1

        # Final operations and fitness computation
        son1.genes = son_genoma_1
        son1.computeFitness()

        son2.genes = son_genoma_2
        son2.computeFitness()

        return son1, son2

    def cycle_crossover(self, indA, indB):

        # Get genes from both parents
        genes_indA = indA.genes
        genes_indB = indB.genes

        # List containing indexes to be changed
        fliping_mating_A = [0]
        first_A = genes_indA[0]
        goner = genes_indB[0]

        while goner != first_A:

            index_new = genes_indA.index(goner)
            goner = genes_indB[index_new]
            fliping_mating_A.append(index_new)

        # Creation of individuals
        son1 = tsp.Individual(self.genSize, self.data)
        son_genoma_1 = genes_indB[:]

        son2 = tsp.Individual(self.genSize, self.data)
        son_genoma_2 = genes_indA[:]

        for i in fliping_mating_A:
            son_genoma_1[i] = genes_indA[i]
            son_genoma_2[i] = genes_indB[i]

        son1.genes = son_genoma_1
        son1.computeFitness()

        son2.genes = son_genoma_2
        son2.computeFitness()

        return son1, son2

    '''
    MUTATION
    '''
    @staticmethod
    def reciprocal_exchange_mutation(ind):

        # Get copy of genes
        genes_ind = ind.genes[:]
        size_ind = ind.genSize

        # Select 2 cities to be mutated randomly
        lista_ale = random.sample(range(0, size_ind), 2)

        # Flip the cities
        flipA, flipB = genes_ind[lista_ale[0]], genes_ind[lista_ale[1]]

        # Insert the corresponding cities
        genes_ind[lista_ale[1]], genes_ind[lista_ale[0]] = flipA, flipB
        ind.genes = genes_ind

        ind.computeFitness()

        return ind

    @staticmethod
    def scramble_mutation(ind):

        # Get copy of genes
        genes_ind = ind.genes[:]
        size_ind = ind.genSize

        # Select a portion of cities to be scrambled
        lista_ale = sorted(random.sample(range(0, size_ind), 2))
        lista_scrumbleable = genes_ind[lista_ale[0]:lista_ale[1]]

        # Randomly, scramble the portion selected
        list_done = random.sample(lista_scrumbleable, len(lista_scrumbleable))

        # Replace the old portion
        genes_ind[lista_ale[0]:lista_ale[1]] = list_done
        ind.genes = genes_ind

        ind.computeFitness()
        return ind

    # Place all individuals to be
    def updateMatingPool(self):

        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

    def newGeneration(self):

        # Calculate number of mutation to be done
        rating_mut = int(self.popSize * self.mutationRate)
        lista_ale = set(random.sample(range(self.popSize/2), rating_mut))

        counter_times = 0

        # Generate n mutations, being n the number of population.
        for i in range(1, round(len(self.population)/ 2)):

            # Select both candidates for mating. Use of different techniques.
            cand1, cand2 = self.operators[2](self)

            # Do the crossover with both candidates. Obtain two offspring.
            desc1, desc2 = self.operators[0](self, cand1, cand2)


            # Check what descendants are to be mutated.
            if i in lista_ale:
                counter_times += 1

                # Mutate both offspring siblings. Apply corresponding mutation
                desc1 = self.operators[1](desc1)
                desc2 = self.operators[1](desc2)

            # Insert in offspring pool
            self.population.append(desc1)
            self.population.append(desc2)

        # Best individuals survival.
        self.delete_worst_pop()

    def delete_worst_pop(self):

        # To order the population pool according to the fitness value. The fitter the best
        lista_ordenada = sorted(self.population, key=lambda k: k.getFitness())


        # Take the 1/3 portion of the list, which implies the number of population
        new_pop = lista_ordenada[0: self.popSize]

        # Update population to be as initial number and the best individual.
        self.population = new_pop
        self.best = self.population[0]

    def GAStep(self):

        self.updateMatingPool()
        self.newGeneration()

    def search(self, file_ex, m_times, m_fitness, tittle):

        self.iteration = 0
        while self.iteration < self.maxIterations:

            ta = time.time()
            self.GAStep()
            tb = time.time()

            if tittle == 2:
                m_times[tittle, self.iteration] = (m_times[tittle, self.iteration] + (tb - ta)) / 3
                m_fitness[tittle, self.iteration] = (m_fitness[tittle, self.iteration] + self.best.getFitness()) / 3
            else:
                m_times[tittle, self.iteration] += (tb - ta)
                m_fitness[tittle, self.iteration] += self.best.getFitness()

            self.iteration += 1

            print("\nIteration #", self.iteration)
            print("Best Solution: ", round(self.best.getFitness(), 2))

            print("Solution: ", self.best.genes)

        print("Total iterations: ", self.iteration)
        file_ex.write('Total iterations: {0}\n'.format(self.iteration))
        print("Best Solution: ", round(self.best.getFitness(), 2))
        file_ex.write('Best solution: {0}\n'.format(self.best.getFitness()))
        return file_ex, self.best.getFitness(), m_times, m_fitness

test_automatization()