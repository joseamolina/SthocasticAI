import random
import os
import numpy as np
import time
import matplotlib.pyplot as plt

class Gsat:

    # Constructor
    def __init__(self, type_used):
        self.set_sols = set()
        # Read the directory
        for root, dirs, files in os.walk("./Inst"):
            # Read the files
            for filename in files:
                # Get the info about the instance
                instance, n_c, n_n = self._read_instances('Inst/{0}'.format(filename))
                # Choose between the Tabu search or the novelty+
                solution = self.solver_tabu(n_c, instance) if type_used == 'Tabu' else self.solver_novelty(n_c, instance)
                # Check if the answer is valid
                print('The solution is {0}'.format(solution[0]))
                self.check_sol(instance, solution[0])

    # It checks if a solution is true or not
    def check_sol(self, instance, solution):

        solution = set(solution)
        val = True
        it = 0

        while val and it != len(instance):

            val = any(i in solution for i in instance[it])
            it += 1

        if val: print('It is a solution.\n\n')
        else: print('It is NOT a solution.\n\n')

    # This method reads the instance and converts it to a list containing each clause as a set.
    def _read_instances(self, inst1):

        print('Reading {0}'.format(inst1))
        set_ins = []
        n_claus = 0
        n_nums = 0

        inst1 = open(inst1, 'r')
        comp_set = set()
        for line1 in inst1.readlines():

            if line1[0] == 'p':
                n_claus, n_nums = line1.split()[2], line1.split()[3]

            elif line1[0] != 'p' and line1[0] != 'c':
                for r in line1.split():
                    if r == '0':
                        if len(comp_set) != 0: set_ins.append(comp_set)
                        comp_set = set()
                    elif r != '%':
                        comp_set.add(int(r))

        return set_ins, int(n_claus), int(n_nums)

    # This method is called when a variable is to be chosen as candidate to be flipped.
    def get_clauses_flipped(self, dic_clauses, var, instance, init, sat_clauses, n_insatistied):

        # For all the clauses that contain that var
        for cla in dic_clauses[abs(var)]:

            # Get the set of the clause
            new_clause = instance[cla]

            # The value satisfies the var in the clause but the clause was not satisfied
            if init in new_clause and not sat_clauses[cla][0]:
                sat_clauses[cla] = (True, 1)
                n_insatistied -= 1

            # The value satisfies the var in the clause and the clause was already satisfied
            elif init in new_clause and sat_clauses[cla][0]:
                sat_clauses[cla] = (True, sat_clauses[cla][1] + 1)

            # The value doesn't satisfy the var in the clause and the clause was already satisfied by more
            # than 1 variable
            elif init not in new_clause and sat_clauses[cla][0] and sat_clauses[cla][1] > 1:
                sat_clauses[cla] = (True, sat_clauses[cla][1] - 1)

            # The value doesn't satisfy the var in the clause and the clause was already satisfied by
            # that variable
            elif init not in new_clause and sat_clauses[cla][0] and sat_clauses[cla][1] == 1:
                sat_clauses[cla] = (False, None)
                n_insatistied += 1

        return sat_clauses, n_insatistied

    # It solves the novelty + algorithm
    def solver_novelty(self, n_vars, instance):

        # Initial parameters
        wp = 0.4
        p = 0.3
        iterations = 100000

        # MOBILE Initial random solution
        init = self.initial_configuration(int(n_vars))

        # MOBILE Get the clauses satisfied/not and the number of unsatisfied clauses
        sat_clauses, n_insatistied = self.satisfaying_clauses(init, instance)

        # STATIC For each var, get the clause where it is
        dic_clauses = self.n_pert(instance)

        most_frequent_flip = [-1 for _ in range(len(sat_clauses))]

        for it in range(iterations):

            if n_insatistied == 0:
                print("Number of iterations used: {0}".format(it))
                return init, it

            # The number of the clause to be treated (FIRST chosen basis)
            cont = 0
            while sat_clauses[cont][0]:
                cont += 1

            # With some probability wp select a random variable from c, while in the remaining
            # cases use Novelty for your variable selection process
            decision = np.random.choice([True, False], 1, p=[wp, 1 - wp])

            # To take a random var of the clause to flip
            if decision[0]:

                # Choose the var randomly
                var = np.random.choice(list(instance[cont]), 1)[0]

                # Get the value of the solution and flip it
                if var not in init:
                    index = init.index(var * -1)
                    init[index] = init[index] * -1

                else:
                    index = init.index(var)
                    init[index] = init[index] * -1

                # change for each clause being this var as the most recently flipped.
                for claus_to_change in dic_clauses[abs(var)]:
                    most_frequent_flip[claus_to_change] = abs(var)

                # Get all clauses where the var flipped in contained
                sat_clauses, n_insatistied = self.get_clauses_flipped(dic_clauses, var, instance, init[index], sat_clauses, n_insatistied)

            # Select the best and the second best
            else:

                # Get the VBest and V2Best vars in the clause
                best_var = 0
                best_second_var = 0
                best_value = float('inf')
                best_second_value = float('inf')
                nr_sat_clas = []
                nr_second_sat_clas = []

                # Loop through the clause to whats the best var to flip
                for var in instance[cont]:

                    # Flip the variable
                    index = init.index(var * -1)
                    to_flip = init[index] * -1

                    # Get a copy of the current solution applyed to the main clause and the
                    #  number of unsatisfied clauses.
                    sat_clauses_new = sat_clauses[:]
                    n_insatistied_new = n_insatistied

                    # Gets the current solution for that flip
                    sat_clauses_new, n_insatistied_new = self.get_clauses_flipped(dic_clauses, var, instance, to_flip,
                                                                                  sat_clauses_new, n_insatistied_new)

                    # If this solution is better than the previous state, we update as the best variable to flip
                    if n_insatistied_new < best_value and n_insatistied_new < best_second_value:
                        best_value = n_insatistied_new
                        best_var = var
                        nr_sat_clas = sat_clauses_new[:]

                    # If this solution is better than the previous state, we update as the second
                    # better variable to flip
                    if best_second_value > n_insatistied_new >= best_value:
                        best_second_value = n_insatistied_new
                        best_second_var = var
                        nr_second_sat_clas = sat_clauses_new[:]

                # If the most frequent flipped var is not the best variable select the best variable of the
                # clause to update
                if most_frequent_flip[cont] != abs(best_var):

                    if best_second_var not in init:
                        index = init.index(best_var * -1)
                        init[index] = init[index] * -1

                    else:
                        index = init.index(best_var)
                        init[index] = init[index] * -1

                    for claus_to_change in dic_clauses[abs(best_var)]:
                        most_frequent_flip[claus_to_change] = abs(best_var)

                    sat_clauses = nr_sat_clas
                    n_insatistied = best_value

                else:

                    # In this case, we choose either to update the second best or the best.
                    best_2 = np.random.choice([True, False], 1, p=[p, 1 - p])
                    if best_2:

                        if best_second_var not in init:
                            index = init.index(best_second_var * -1)
                            init[index] = init[index] * -1

                        else:
                            index = init.index(best_second_var)
                            init[index] = init[index] * -1

                        for claus_to_change in dic_clauses[abs(best_second_var)]:
                            most_frequent_flip[claus_to_change] = abs(best_second_var)
                        sat_clauses = nr_second_sat_clas
                        n_insatistied = best_second_value

                    else:
                        if best_var not in init:
                            index = init.index(best_var * -1)
                            init[index] = init[index] * -1

                        else:
                            index = init.index(best_var)
                            init[index] = init[index] * -1

                        for claus_to_change in dic_clauses[abs(best_var)]:
                            most_frequent_flip[claus_to_change] = abs(best_var)
                        sat_clauses = nr_sat_clas
                        n_insatistied = best_value

    def solver_tabu(self, n_clauses, instance):

        # Hyper parameters
        n_restarts = 10
        iterations = 1000
        tl = 5

        while n_restarts > 0:

            # Gets an initial random configuration
            init = self.initial_configuration(int(n_clauses))
            # gets the dictionary of the satisfying clauses and the number of clauses satisfied
            sat_clauses, n_insatistied = self.satisfaying_clauses(init, instance)

            # Gets the dictionary of all vars
            dic_clauses = self.n_pert(instance)

            # Tabu dictionary: K: Var, V: n_iterations
            tabu_dic = {}

            # For each eteration
            for it in range(iterations):

                # If all clauses are satisfied, we have finished.
                if n_insatistied == 0:
                    print("Number of iterations used: {0}".format(it))
                    return init, it

                # Increase the number of steps for each var in the tabu dictionary.
                # If the number of steps is overcome, we delete it from the dic.
                delete_vars = None
                for var_dic in tabu_dic:
                    tabu_dic[var_dic] = tabu_dic[var_dic] + 1

                    if tabu_dic[var_dic] == tl:
                        delete_vars = var_dic

                if delete_vars is not None:
                    del tabu_dic[delete_vars]

                best_var = 0
                best_value = float('inf')
                nr_sat_clas = []

                # For each var of the current solution
                for var in init:

                    # Check if the var is contained in the tabu_dic
                    if not abs(var) in tabu_dic:

                        # Flip the var
                        to_flip = var
                        to_flip = to_flip * -1

                        # Get a copy of the current solution applyed to
                        # the main clause and the number of unsatisfied clauses.
                        sat_clauses_new = sat_clauses[:]
                        n_insatistied_new = n_insatistied

                        # Gets the current solution for that flip
                        sat_clauses_new, n_insatistied_new = self.get_clauses_flipped(dic_clauses, var, instance,
                                                                                      to_flip,
                                                                                      sat_clauses_new,
                                                                                      n_insatistied_new)
                        # If this solution is better than the previous state, we update as the best variable to flip
                        if n_insatistied_new < best_value:
                            best_value = n_insatistied_new
                            best_var = var
                            nr_sat_clas = sat_clauses_new

                # When all variables are checked, we flip to the solution the best variable.
                # We update the tabu dictionary.
                index = init.index(best_var)
                init[index] = init[index] * -1
                n_insatistied = best_value
                tabu_dic[abs(best_var)] = 0
                sat_clauses = nr_sat_clas

    # It returns an initial random solution
    def initial_configuration(self, n_clauses):
        return list(map(lambda r: random.choice([r, r * -1]), [i for i in range(1, n_clauses + 1)]))

    # It calculates the clauses satisfied or not AND the number of unsatisfied clauses
    def satisfaying_clauses(self, conf, instance):

        sol = set(conf)
        cost = []
        n_ins = 0
        for clause in instance:

            n_vars = 0
            for i in clause:

                if i in sol:
                    n_vars += 1

            if n_vars == 0:
                cost.append((False, None))
                n_ins += 1
            else:
                cost.append((True, n_vars))
        return cost, n_ins

    # It produces a dictionary per each variable containing the clauses that is contained in.
    # It is used for knowing where each variable is contained.
    @staticmethod
    def n_pert(instances):

        dic_clauses = {}
        for num, ins in enumerate(instances):

            for var in ins:

                if abs(var) in dic_clauses:
                    enim = dic_clauses[abs(var)]
                    enim.append(num)
                    dic_clauses[abs(var)] = enim
                else:
                    dic_clauses[abs(var)] = [num]

        return dic_clauses

    # It does the RTD charts
    def test_algorithms(self):

        # Triggers Las Vegas stochastic algorithm
        n_executions = 100
        for filename in ['uf20-020.cnf', 'uf20-021.cnf']:

            for type in [self.solver_novelty, self.solver_tabu]:

                rt_e = []
                for i in range(n_executions):
                    instance, n_c, n_n = self._read_instances('Inst/{0}'.format(filename))
                    t1 = time.time()
                    # Call the algorithm for the given instance
                    type(n_c, instance)
                    t2 = time.time()
                    rt_e.append(t2 - t1)

                # Sorts the algorithm
                rt_e = sorted(rt_e)
                sum_total = sum(rt_e)
                # It applyes the regular division for the probability
                rt_e = list(map(lambda t: (t * 1)/sum_total, rt_e))

                # It plot the results
                plt.plot(rt_e)
                plt.ylabel('P(run time)')
                plt.xlabel('Executions')
                plt.title('{0}, with {1}'.format(filename, type.__name__))
                plt.show()

    def get_chart_algorithms(self):

        solv_t = []
        solv_n = []

        for (root, dirs, files) in os.walk("./Inst"):

            n_executions = 1000

            for filename in files:
                print(files)

                it_n = 0
                it_t = 0
                for _ in range(n_executions):

                    # Get the info about the instance
                    instance, n_c, n_n = self._read_instances('Inst/{0}'.format(filename))
                    # Choose between the Tabu search or the novelty+

                    solution_t = self.solver_tabu(n_c, instance)
                    it_t += solution_t[1]

                    solution_n = self.solver_novelty(n_c, instance)
                    it_n += solution_n[1]

                it_n /= n_executions
                it_t /= n_executions
                solv_n.append(it_n)
                solv_t.append(it_t)

            plt.xlabel("Instance")
            plt.ylabel("Average of number of executions per algorithm")
            plt.title('Number of executions as per 100 average for Novelty+ and Tabu')
            plt.plot(solv_t, color='red')
            plt.plot(solv_n, color='blue')
            plt.xticks(np.arange(9), files)
            plt.show()

obj = Gsat('novelty')
# Choose the method to execute in order to get different chart results.
#obj.get_chart_algorithms()
obj.test_algorithms()
