import random
import time
import numpy as np
from scipy.spatial.distance import euclidean

class Project_tsp_jose:

    # This method reads the data set and returns a list of nodes,
    # a dictionary of nodes, and the number of nodes.
    @staticmethod
    def read_data_set(file_name):
        try:
            n_nodes = 0
            file_instance = open(file_name, 'r')
            list_nodes = {}

            for c, line in enumerate(file_instance, 0):
                if c == 0:
                    n_nodes = int(line)
                else:
                    list_line = line.split(' ')
                    list_nodes[int(list_line[0])] = [int(list_line[1]), int(list_line[2])]

            list_edges = [i for i in range(1, n_nodes + 1)]
            return n_nodes, list_nodes, list_edges

        except FileNotFoundError:
            print('That file does not exist')
            exit(-1)

    # Totally initial random solution
    @staticmethod
    def get_random_sol(list_edges):

        list_shuffled = list_edges[:]
        random.shuffle(list_shuffled)

        return list_shuffled

    # Gets a KNN Initial solution
    def get_knn_sol(self, list_nodes, dic_nodes):

        initial = random.choice(list_nodes)
        done = set(list_nodes)
        tour = []

        while len(tour) != len(list_nodes):

            best = 0
            dis_best = float('inf')
            for node in done:
                distance = self.get_distance(initial, node, dic_nodes)
                if distance < dis_best:
                    best = node
                    dis_best = distance

            tour.append(best)
            done.remove(best)

        return tour

    # It gives the following node in the tour
    @staticmethod
    def get_next_edge(index_node, solution):

        if index_node == len(solution) - 1:
            return solution[1], 1
        else:
            return solution[index_node + 1], index_node + 1

    # Euclidean distance between 2 nodes
    @staticmethod
    def get_distance(node0, node1, dic_nodes):

        return euclidean(dic_nodes[node0], dic_nodes[node1])

    # Does perturbation stage
    @staticmethod
    def perturbation(solution):

        sol_perturbed = solution[:]

        for time_rep in range(5):

            nodes = sorted(random.sample(range(len(solution)), 2))

            a, c = nodes[0], nodes[1]
            b, d = a + 1, c + 1

            sol_perturbed = sol_perturbed[:a + 1] + sol_perturbed[b:d][::-1] + sol_perturbed[d:]

        return sol_perturbed

    # It calculates the whole distance of the tour
    def calculate_distance(self, tour, dic_nodes):

        new_tour_distance = 0
        for node in range(len(tour) - 1):
            new_tour_distance += self.get_distance(tour[node], tour[node + 1], dic_nodes)

        new_tour_distance += self.get_distance(tour[len(tour) - 1], tour[0], dic_nodes)

        return new_tour_distance

    # It decides whether to update s_sigma or s_plus
    @staticmethod
    def accept_criteria(s_sigma_pls, s_sigma_pls_distance, s_plus, s_plus_distance):

        # With some probability, decide what to do
        wp = 0.05
        decision = np.random.choice([True, False], 1, p=[wp, 1 - wp])

        if decision:
            return s_sigma_pls, s_sigma_pls_distance
        else:

            if s_sigma_pls_distance < s_plus_distance:
                return s_sigma_pls, s_sigma_pls_distance
            else:
                return s_plus, s_plus_distance

    # It resolves the TSP algorithm. You should call it for resolving it
    def iterative_local_search(self, knn_random, filename, iterations):

        iterations_cost = []
        execution_time = 5

        # Get an initial solution
        n_nodes, dic_nodes, list_edges = self.read_data_set(filename)
        s0 = self.get_random_sol(list_edges) if knn_random == 'knn' else self.get_knn_sol(list_edges, dic_nodes)

        # First local search. It reduces drastically the distance. First approach.
        s_plus_distance, s_plus = self.local_search(n_nodes, s0, dic_nodes, iterations)

        time1 = time.time()

        time2 = time.time()

        # Execution loop. It lasts 5 minutes. It escapes from local minimum and attempts to
        # search a better solution.
        while (time2 - time1) < (execution_time * 60):

            # It perturbs the current solution trying to shuffle the solution.
            s_sigma = self.perturbation(s_plus)

            # It reduces the perturbation to try to find the absolute minimum
            s_sigma_pls_distance, s_sigma_pls = self.local_search(n_nodes, s_sigma, dic_nodes, iterations)

            # It accept what solution to update. The perturbed or the current solution.
            s_plus, s_plus_distance = self.accept_criteria(s_sigma_pls, s_sigma_pls_distance, s_plus, s_plus_distance)

            iterations_cost.append(s_plus_distance)
            time2 = time.time()

        print('The solution with {0} and {1} is {2}'.format(knn_random, filename, s_plus))
        print('The cost of the travel is {0}'.format(s_plus_distance))

        return iterations_cost

    def local_search(self, n_cities, tour, dic_nodes, iterations):

        # Generate all combinations. We select the edge number
        genered_tour = tour

        # It gets the distance of the initial solution
        genered_distance = self.calculate_distance(tour, dic_nodes)

        iterations.append(genered_distance)


        # It iterates until it cannot find a better solution
        is_better = True
        while is_better:

            # It gets randomly an edge (A)
            edge_a = random.sample(range(n_cities), 1)[0]

            # It gets edge (B)
            edge_b = edge_a + 1
            counter_b = 0

            min_local = genered_distance
            local_tour = genered_tour[:]

            if edge_b >= len(tour):
                edge_b = 0

            # It iterates the number of positions for edge B
            for _ in range(len(tour) - 2):

                if edge_b >= len(tour):
                    edge_b = 0

                # It gets node (C)
                edge_c = edge_b + 1

                if edge_c >= len(tour):
                    edge_c = 0

                # It iterates the positions for edge C
                for _ in range(len(tour) - 2 - counter_b):

                    if edge_c >= len(tour):
                        edge_c = 0

                    # It gets the initial node of each edge.
                    a, c, e = edge_a, edge_b, edge_c

                    # It sorts the bunch of nodes.
                    it_new = sorted([a, c, e])
                    a, c, e = it_new[0], it_new[1], it_new[2]

                    # It gets the seconds nodes for each edge.
                    b = a + 1
                    d = c + 1
                    f = e + 1

                    # Obtain the corresponding values of the tour given the indexes
                    a_val, b_val = genered_tour[a], genered_tour[b]
                    c_val, d_val = genered_tour[c], genered_tour[d]
                    e_val = genered_tour[e]
                    f_val = genered_tour[0] if f >= len(tour) else genered_tour[f]

                    # Initial distance to be computed of the current state of the 3 links
                    # CURRENT COMBINATION: a-b, c-d, e-f
                    dis_gen = self.get_distance(a_val, b_val, dic_nodes) + self.get_distance(c_val, d_val, dic_nodes) + self.get_distance(e_val, f_val, dic_nodes)

                    # So far, the best distance is the current distance
                    best_dis = dis_gen
                    candidate = None

                    # If a new combination of 3 links is better than the current, we would change.
                    # NEW COMBINATION:
                    dis1 = self.get_distance(a_val, c_val, dic_nodes) + self.get_distance(b_val, d_val, dic_nodes) + self.get_distance(e_val, f_val, dic_nodes)
                    if dis1 < best_dis:
                        best_dis = dis1
                        candidate = 1

                    # NEW COMBINATION:
                    dis2 = self.get_distance(a_val, e_val, dic_nodes) + self.get_distance(b_val, f_val, dic_nodes) + self.get_distance(c_val, d_val, dic_nodes)
                    if dis2 < best_dis:
                        best_dis = dis2
                        candidate = 2

                    # NEW COMBINATION:
                    dis3 = self.get_distance(c_val, e_val, dic_nodes) + self.get_distance(d_val, f_val, dic_nodes) + self.get_distance(a_val, b_val, dic_nodes)
                    if dis3 < best_dis:
                        best_dis = dis3
                        candidate = 3

                    # NEW COMBINATION:
                    dis4 = self.get_distance(a_val, c_val, dic_nodes) + self.get_distance(b_val, e_val, dic_nodes) + self.get_distance(d_val, f_val, dic_nodes)
                    if dis4 < best_dis:
                        best_dis = dis4
                        candidate = 4

                    # NEW COMBINATION:
                    dis5 = self.get_distance(a_val, e_val, dic_nodes) + self.get_distance(b_val, d_val, dic_nodes) + self.get_distance(c_val, f_val, dic_nodes)
                    if dis5 < best_dis:
                        best_dis = dis5
                        candidate = 5

                    # NEW COMBINATION:
                    dis6 = self.get_distance(a_val, d_val, dic_nodes) + self.get_distance(b_val, f_val, dic_nodes) + self.get_distance(c_val, e_val, dic_nodes)
                    if dis6 < best_dis:
                        best_dis = dis6
                        candidate = 6

                    # NEW COMBINATION:
                    dis7 = self.get_distance(a_val, d_val, dic_nodes) + self.get_distance(b_val, e_val, dic_nodes) + self.get_distance(c_val, f_val, dic_nodes)
                    if dis7 < best_dis:
                        best_dis = dis7
                        candidate = 7

                    # If these changes are better than the ones done
                    if candidate is not None:

                        # Calculate distance therefore generated.
                        new_dis = genered_distance - dis_gen + best_dis

                        # Keep the candidate to do the movement.
                        if new_dis < min_local:
                            min_local = new_dis
                            local_tour = (candidate, (a, b, c, d, e, f))

                    edge_c += 1

                edge_b += 1
                counter_b += 1

            # At this point, we should check if any movement would make the tour shorter than it was before or not.
            if genered_distance <= min_local:
                is_better = False
            else:

                # Get the combination of nodes to update.
                candidate, nodes = local_tour
                a, b, c, d, e, f = nodes

                # Depending on the combination, apply a series of changes to the route.
                first_frac = genered_tour[:a + 1]
                last_frac = genered_tour[f:]

                if candidate == 1:
                    gen = genered_tour[b:d][::-1]
                    d_tour = genered_tour[d:]
                    best_tour = first_frac + gen + d_tour

                elif candidate == 2:
                    gen = genered_tour[b:f][::-1]
                    best_tour = first_frac + gen + last_frac

                elif candidate == 3:
                    c_tour = genered_tour[:c + 1]
                    gen = genered_tour[d:f][::-1]
                    best_tour = c_tour + gen + last_frac

                elif candidate == 4:
                    gen = genered_tour[b:c + 1][::-1] + genered_tour[c + 1:f][::-1]
                    best_tour = first_frac + gen + last_frac

                elif candidate == 5:
                    gen = genered_tour[c + 1:f][::-1] + genered_tour[b:c + 1]
                    best_tour = first_frac + gen + last_frac

                elif candidate == 6:
                    gen = genered_tour[c + 1:f] + genered_tour[b:c + 1][::-1]
                    best_tour = first_frac + gen + last_frac

                elif candidate == 7:
                    gen = genered_tour[c + 1:f] + genered_tour[b:c + 1]
                    best_tour = first_frac + gen + last_frac

                # Update the tour to make it the current tour.
                genered_distance = min_local
                genered_tour = best_tour
                iterations.append(genered_distance)

        iterations.append(genered_distance)

        return genered_distance, genered_tour

ins = Project_tsp_jose()
for heur in ['knn', 'random']:
    for inst in ['dataset/inst-0.tsp', 'dataset/inst-13.tsp']:

        import matplotlib.pyplot as plt
        for its in range(5):

            iterations = []
            ins.iterative_local_search(heur, inst, iterations)

            plt.plot(iterations)
            plt.ylabel('Cost')
            plt.xlabel('Reduction')
            plt.title('{0}, with {1}'.format(heur, inst))
        plt.show()