import copy
import multiprocessing
import os
import time

from Helper import *
import numpy as np
import json
from shutil import copyfile

from Optimizer import Optimizer
from plot import plot_errs_final, plot_avg_errs_final, plot_dist


class DifferentialOptimizer(Optimizer):
    def __init__(self, pop_size, max_gen, cr=0.9, f=0.8, log=True, plot=False, file_lst=None, repeats=15,
                 overhead_fac=0.2, avg_error_fac=0.4, clean_deg_len_fac=-0.0001, clean_avg_error_fac=0.2, chunksize=50,
                 initialize=True, non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, store_state_foldername=None,
                 seed_spacing=0, use_payload_xor=False):

        super().__init__(pop_size, max_gen, log, plot, file_lst, repeats, overhead_fac, avg_error_fac,
                         clean_deg_len_fac, clean_avg_error_fac, non_unique_packets_fac, unrecovered_packets_fac,
                         chunksize, initialize, store_state_foldername=store_state_foldername,
                         seed_spacing=seed_spacing, use_payload_xor=use_payload_xor)
        self.f = f
        self.cr = cr
        self.np_random_state = np.random.Generator(np.random.PCG64())
        try:
            if log:
                self.dir = f"{self.store_state_foldername}/DffEv_" + str(pop_size) + "_" + str(max_gen) + "_" + \
                           str(self.cr).replace(".", "") + "_" + str(self.f).replace(".", "")
                self.file = self.dir + "/"
                self.file_con = list()
                os.makedirs(self.dir, exist_ok=True)
        except FileExistsError:
            print("Dir already exists. Using it anyway.")

    def signal_handler(self, sig, frame):
        print('Storing State...')
        self.store_state()
        # sys.exit(0)

    def get_state(self):
        state = super().get_state()
        state["cr"] = self.cr
        state["f"] = self.f
        return state

    def store_state(self, filename=None):
        if filename is None:
            filename = f"{self.store_state_foldername}/diff_opt_state_" + str(self.finished_gen) + ".json"
        super().store_state(self.get_state(), filename)
        copyfile(filename, f"{self.store_state_foldername}/diff_opt_state.json")

    @staticmethod
    def load_from_state(filename):
        with open(filename, "r") as fp:
            state = json.load(fp)
        pop_size = state.get("pop_size")
        max_gen = state.get("max_gen")
        cr = state.get("cr")
        f = state.get("f")
        log = state.get("log")
        plot = state.get("plot")
        file_lst = state.get("file_lst")
        repeats = state.get("repeats")
        overhead_fac = state.get("overhead_fac")
        avg_error_fac = state.get("avg_error_fac")
        clean_deg_len_fac = state.get("clean_deg_len_fac")
        clean_avg_error_fac = state.get("clean_avg_error_fac")
        non_unique_packets_fac = state.get("non_unique_packets_fac")
        unrecovered_packets_fac = state.get("unrecovered_packets_fac")
        use_payload_xor = state.get("use_payload_xor")
        seed_spacing = state.get("seed_spacing")

        chunksize = state.get("chunksize")
        pop = [Distribution.Distribution.from_json(x) for x in state.get("pop")]
        finished_gen = state.get("finished_gen")
        finished_prev_best = Distribution.Distribution.from_json(state.get("finished_prev_best"))
        finished_rungs_wo_imprv = state.get("finished_runs_wo_imprv")
        tmp = DifferentialOptimizer(pop_size=pop_size, max_gen=max_gen, cr=cr, f=f,
                                    log=log, plot=plot, file_lst=file_lst, repeats=repeats, overhead_fac=overhead_fac,
                                    avg_error_fac=avg_error_fac, clean_deg_len_fac=clean_deg_len_fac,
                                    clean_avg_error_fac=clean_avg_error_fac,
                                    non_unique_packets_fac=non_unique_packets_fac,
                                    unrecovered_packets_fac=unrecovered_packets_fac, chunksize=chunksize,
                                    initialize=False, seed_spacing=seed_spacing, use_payload_xor=use_payload_xor)
        tmp.gen_best_dist = [Distribution.Distribution.from_json(x) for x in state.get("gen_best_dist")]
        tmp.gen_avg_err = state.get("gen_avg_err")
        tmp.gen_avg_over = state.get("gen_avg_over")
        tmp.gen_calculated_error = state.get("gen_calculated_error")
        tmp.err_fit = state.get("err_fit")
        tmp.calc_err_fit = state.get("calc_err_fit")
        if tmp.calc_err_fit is not None:
            tmp.calc_err_fit = np.poly1d(tmp.calc_err_fit)
        tmp.pop = pop
        tmp.finished_gen = finished_gen
        tmp.finish_prev_best = finished_prev_best
        tmp.finished_runs_wo_imprv = finished_rungs_wo_imprv
        return tmp

    def optimize(self, start_gen=0):
        """
        Runs max_gen iterations or breaks, if 250 iterations without any improvements were performed and returns the
        best distribution after finishing.
        :return:
        """
        prev_best = self.finish_prev_best
        runs_wo_imprv = self.finished_runs_wo_imprv
        for gen in range(start_gen, self.max_gen):
            print("########## Generation " + str(gen) + "/" + str(self.max_gen) + ". ##########")
            self.signal_handler(0, 0)
            if self.log:
                gen_list = [gen]
                for d in self.pop:
                    gen_list.append((
                        d.avg_err, d.overhead, d.dist_lst, d.non_unique_packets, d.degree_errs, d.clean_deg_len,
                        d.clean_avg_error))
                self.file_con.append(gen_list)
            self.pop = self.compute_generation_diff()
            sorted_dists = sorted(self.pop, key=lambda x: x.calculate_error_value())
            # avg_err + (x.overhead * self.overhead_fac))
            prev_best, runs_wo_imprv = self.select_best(prev_best, runs_wo_imprv, sorted_dists)
            if runs_wo_imprv >= 250:
                break
            self.gen_best_dist.append(copy.deepcopy(sorted_dists[0]))
            self.gen_avg_err.append(sum([d.avg_err for d in self.pop]) / self.pop_size)
            self.gen_avg_over.append(sum([d.overhead for d in self.pop]) / self.pop_size)
            self.gen_calculated_error.append(sum([d.calculate_error_value() for d in self.pop]) / self.pop_size)
            self.gen_clean_avg_err.append(sum([d.clean_avg_error for d in self.pop]) / self.pop_size)
            if self.plot and gen % 10 == 0 and gen != 0:
                # plot_errs_final(self.gen_best_dist save=True, name=None)
                plot_errs_final(self.gen_best_dist, save=True, name=self.file + "best_results_" + str(gen))
                plot_avg_errs_final(self.gen_clean_avg_err, self.gen_calculated_error, save=True,
                                    name=self.file + "clean_average_results_" + str(gen))
                plot_avg_errs_final(self.gen_avg_err, self.gen_calculated_error, save=True,
                                    name=self.file + "average_results_" + str(gen))
                save_to_csv(self.file_con, self.file + "optimization_log_" + str(gen))
            self.finished_gen = gen
            self.finished_runs_wo_imprv = runs_wo_imprv
            self.finish_prev_best = prev_best
        self.signal_handler(0, 0)
        if self.log:
            prev_best.save_to_txt(self.file + "best_dist_" + str(int(time.time())))
            plot_errs_final(self.gen_best_dist, save=True, name=self.file + "best_results")
            self.err_fit, self.calc_err_fit, self.gens = plot_avg_errs_final(self.gen_avg_err,
                                                                             self.gen_calculated_error,
                                                                             save=True,
                                                                             name=self.file + "average_results")
            save_to_csv(self.file_con, self.file + "optimization_log")
        return prev_best

    def select_best(self, prev_best, runs_wo_imprv, selected_dists):
        """
        Selects the best distribution of the current population and plots it, if it's better than the previous best.
        Adds 1 to runs_wo_imprv if no improvement were made.
        :param prev_best:
        :param runs_wo_imprv:
        :param selected_dists:
        :return:
        """
        if prev_best is not None:
            if self.plot:
                plot_dist(prev_best, True, name=self.file + "dff_best_results_select_best_" + str(self.finished_gen))
            if prev_best.calculate_error_value() > selected_dists[0].calculate_error_value():
                prev_best = copy.deepcopy(selected_dists[0])
                runs_wo_imprv = 0
            else:
                print("----- Optimal distribution has not changed. -----")
                runs_wo_imprv += 1
            print("Generations best synthetic error value: " + str(
                round(selected_dists[0].calculate_error_value(), 4)))
        else:
            prev_best = copy.deepcopy(selected_dists[0])
            if self.plot:
                plot_dist(prev_best, True, name=self.file + "_dff_best_results_select_best_" + str(self.finished_gen))
        return prev_best, runs_wo_imprv

    def compute_generation_diff(self):
        """
        Multiprocessing wrapper method for @inner_compute_generation_diff which computes the new generation following
        the differential evolution algorithm.
        :return:
        """
        p = multiprocessing.Pool(self.cores)
        next_gen = p.map(self.inner_compute_generation_diff_custom, self.pop)
        p.close()
        try:
            p.join()
        except Exception as ex:
            print("error while join()'ing with timeout", ex)
            print(next_gen)
        return next_gen

    def inner_compute_generation_diff(self, dist):
        """
        Implementation of the recombination algorithm of the differential evolution.
        Computes a new distribution for the given one.
        :param dist:
        :return:
        """
        new_dist_lst = copy.deepcopy(dist.dist_lst)
        # choose 3 distributions (a,b and c) from the population (with all !=  dist )
        tmp_dists = [d for d in self.pop if d.dist_lst != dist.dist_lst]
        ind = self.np_random_state.choice(range(len(tmp_dists)), size=3)
        # choose R as a random degree of the distribution
        r = self.np_random_state.choice(range(len(new_dist_lst)))
        comp_dists = [tmp_dists[x] for x in ind]
        prob_lst = self.np_random_state.uniform(0.0, 1.0, size=len(new_dist_lst))
        for i in range(0, len(prob_lst)):
            # if random number < crossover rate OR _i_ is the random degree: calculate crossover
            if prob_lst[i] < self.cr or i == r:
                new_dist_lst[i] = comp_dists[0].dist_lst[i] + self.f * (
                        comp_dists[1].dist_lst[i] - comp_dists[2].dist_lst[i])
                # limit to 0
                if new_dist_lst[i] < 0:
                    new_dist_lst[i] = 0
            # ensure that degree 1 is not 0 ( to allow successful decoding in any case! )
        if new_dist_lst[0] < 0.05:
            new_dist_lst[0] = 0.05
        new_dist = Distribution.Distribution(norm_list(new_dist_lst), overhead_fac=self.overhead_fac,
                                             avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                             clean_avg_error_fac=self.clean_avg_error_fac,
                                             seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        new_dist.compute_fitness(self.file_lst, repeats=self.repeats, chunksize=self.chunksize)
        if new_dist.calculate_error_value() < dist.calculate_error_value():
            return new_dist
        else:
            return dist

    def inner_compute_generation_diff_custom(self, dist: Distribution.Distribution):
        """
        Custom recombination approach based on per-degree error value compared to the average error level
        :param dist:
        :return:
        """
        new_dist_lst = copy.deepcopy(dist.dist_lst)
        # choose 3 distributions (a,b and c) from the population (with all !=  dist )
        tmp_dists = [d for d in self.pop if d.dist_lst != dist.dist_lst]
        ind = self.np_random_state.choice(range(len(tmp_dists)), size=3)
        # choose R as a random degree of the distribution
        r = self.np_random_state.choice(range(len(new_dist_lst)))
        comp_dists = [tmp_dists[x] for x in ind]
        prob_lst = self.np_random_state.uniform(0.0, 1.0, size=len(new_dist_lst))
        for i in range(0, len(prob_lst)):
            # if random number < crossover rate OR _i_ is the random degree: change probability based on the rel. error
            if prob_lst[i] < self.cr or i == r:
                # to_choose = np.argmin([d.degree_errs[i] for d in comp_dists])
                # scale the degree prob according to the ratio between avg and this degree error value
                # new_dist_lst[i] = comp_dists[to_choose].dist_lst[i] * (
                #        comp_dists[to_choose].avg_err / comp_dists[to_choose].degree_errs[i])
                new_dist_lst[i] = comp_dists[0].dist_lst[i] + self.f * (
                        comp_dists[1].dist_lst[i] - comp_dists[2].dist_lst[i])
                # limit to 0
                if new_dist_lst[i] < 0:
                    new_dist_lst[i] = 0
        # ensure that degree 1 is not 0 ( to allow successful decoding in any case! )
        # if new_dist_lst[0] < 0.05:
        #    new_dist_lst[0] = 0.05
        new_dist = Distribution.Distribution(norm_list(new_dist_lst), overhead_fac=self.overhead_fac,
                                             avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                             clean_avg_error_fac=self.clean_avg_error_fac,
                                             seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        new_dist.compute_fitness(self.file_lst, repeats=self.repeats, chunksize=self.chunksize)
        if new_dist.calculate_error_value() <= dist.calculate_error_value():
            return new_dist
        else:
            return dist
