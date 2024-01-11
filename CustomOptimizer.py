import multiprocessing
from Helper import *
import matplotlib.pyplot as plt

from plot import plot_dist


class CustomOptimizer():
    def __init__(self, max_grades=8, file_lst=None, log=False, repeats=15, plot=False, overhead_fac=0.5, chunksize=50):
        self.file_lst = file_lst
        self.max_grades = max_grades
        self.log = log
        self.repeats = repeats
        self.plot = plot
        self.chunksize = chunksize
        self.overhead_fac = overhead_fac
        self.errs = list()
        self.over = list()
        self.cores = multiprocessing.cpu_count() - 1
        if self.file_lst is None:
            self.file_lst = ['Dorn']
        self.compute_grade_dists()

    def compute_grade_dists(self, grades=40):
        """
        Creates a distribution for every grade that has 5% probability for grade 1 and 95% probability for the other
        grade (1-40). Returns a list with the avg_errs and a list containing the overhead per grade.
        :param grades: Number of grades for the distributions.
        :return:
        """
        dists = list()
        for x in range(0, 40):
            dist_lst = [0 if i != x else 0.95 for i in range(0, grades)]
            dist_lst[0] += 0.05
            dists.append(Distribution.Distribution(dist_lst))
        p = multiprocessing.Pool(self.cores)
        dists = p.map(self.inner_compute_fitness, dists)
        p.close()
        for x in range(0, grades):
            self.errs.append(dists[x].degree_errs[x])
            self.over.append(dists[x].overhead)
        if self.plot:
            self.plot_grade_err()

    def plot_grade_err(self, grades=40):
        """
        Method to plot the degree-wise error probabilities and overheads.
        :return:
        """
        err_plt = plt.plot([x for x in range(1, grades + 1)], self.errs, 'k', label='average error')
        plt.ylabel('average error')
        plt.xlabel('Grad')
        plt.grid()
        plt.twinx()
        over_plt = plt.plot([x for x in range(1, grades + 1)], self.over, 'r', label='average Overhead')
        plt.ylabel('average Overhead')
        lns = err_plt + over_plt
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='best')
        plt.savefig("cust_opt_" + self.file_lst[0] + ".svg", format='svg', dpi=1200)
        plt.show()

    def inner_compute_fitness(self, dist):
        """
        Helper method to compute the distribution fitness using multiprocessing.
        :param dist: Distribution to compute the fitness for.
        :return:
        """
        dist.compute_fitness(self.file_lst, self.repeats, self.chunksize)
        return dist

    def create_dist_from_res(self, grade_one_prob=0.05, grades=4, use_overhead=True):
        """
        Creates a distribution based on the computed avg_errs per grade and the parameters given.
        :param grade_one_prob: The probability of the first grade.
        :param grades: Number of best grades to use.
        :param use_overhead:
        :return:
        """
        if use_overhead:
            comp_list = [self.errs[x] + (self.over[x] * self.overhead_fac) for x in range(len(self.errs))]
        else:
            comp_list = self.errs
        lowest = sorted(comp_list)[:grades]
        indices = list()
        for val in lowest:
            indices.append(comp_list.index(val))
        dist_lst = [1.0 / grades if x in indices else 0.0 for x in range(0, 40)]
        if dist_lst[0] == 0.0:
            dist_lst[0] += grade_one_prob
        return Distribution.Distribution(norm_list(dist_lst), overhead_fac=self.overhead_fac,
                                         avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                         clean_avg_error_fac=self.clean_avg_error_fac,
                                         non_unique_packets_fac=self.non_unique_packets_fac,
                                         unrecovered_packets_fac=self.unrecovered_packets_fac)

    def optimize(self):
        """
        Creates n distributions where n is the number of maximum grades to use. Each grade gets the same probability and
        the first grade will be present in all distributions to make sure decoding will be possible.
        :return:
        """
        dists = list()
        for grade in range(1, self.max_grades + 1):
            dist = self.create_dist_from_res(grades=grade)
            dists.append(dist)
        p = multiprocessing.Pool(self.cores)
        dists = p.map(self.inner_compute_fitness, dists)
        p.close()
        if self.plot:
            [plot_dist(dist) for dist in dists]
        best = select_distributions(dists, 1, overhead_fac=self.overhead_fac)[0]
        plot_dist(best, save=True, name="cust_opt_" + self.file_lst[0] + "_best")
        best.save_to_txt("cust_opt_" + self.file_lst[0] + "_best")
        return best
