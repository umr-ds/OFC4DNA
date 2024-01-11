import os
import copy
import json
from shutil import copyfile
import multiprocessing
from Helper import *

from Optimizer import Optimizer
from plot import plot_errs_final, plot_avg_errs_final, plot_dist


class EvolutionaryOptimizer(Optimizer):
    def __init__(self, pop_size, max_gen, mut_rate=0.05, over_fit=None, file_lst=None, log=True, overhead_fac=0.2,
                 repeats=15, plot=False, avg_error_fac=0.4, clean_deg_len_fac=-0.0001, clean_avg_error_fac=0.2,
                 non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, chunksize=50, initialize=True,
                 store_state_foldername=None, seed_spacing=0, use_payload_xor=False):
        super().__init__(pop_size, max_gen, log, plot, file_lst, repeats, overhead_fac, avg_error_fac,
                         clean_deg_len_fac, clean_avg_error_fac, non_unique_packets_fac, unrecovered_packets_fac,
                         chunksize, initialize, store_state_foldername=store_state_foldername,
                         seed_spacing=seed_spacing, use_payload_xor=use_payload_xor)

        self.mut_rate = mut_rate
        self.over_fit = over_fit
        try:
            if log:
                self.dir = f"{self.store_state_foldername}/EvAlg_" + str(pop_size) + "_" + str(max_gen) + "_" + str(
                    mut_rate)
                self.file = self.dir + "/"
                self.file_con = list()
                os.makedirs(self.dir, exist_ok=True)
        except FileExistsError:
            print("Dir already exists. Using it anyway.")

    def store_state(self, filename=None):
        if filename is None:
            filename = f"{self.store_state_foldername}/evo_opt_state_" + str(self.finished_gen) + ".json"
        super().store_state(self.get_state(), filename)
        copyfile(filename, f"{self.store_state_foldername}/evo_opt_state.json")

    def signal_handler(self, sig, frame):
        print('Storing State...')
        self.store_state()
        # sys.exit(0)

    def get_state(self):
        state = super().get_state()
        state["mut_rate"] = self.mut_rate
        state["over_fit"] = self.over_fit
        return state

    @staticmethod
    def load_from_state(filename):
        with open(filename, "r") as fp:
            state = json.load(fp)
        pop_size = state.get("pop_size")
        max_gen = state.get("max_gen")
        mut_rate = state.get("mut_rate")
        over_fit = state.get("over_fit")
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

        chunksize = state.get("chunksize")
        pop = [Distribution.Distribution.from_json(x) for x in state.get("pop")]
        finished_gen = state.get("finished_gen")
        finished_prev_best = Distribution.Distribution.from_json(state.get("finished_prev_best"))
        finished_rungs_wo_imprv = state.get("finished_runs_wo_imprv")
        tmp = EvolutionaryOptimizer(pop_size=pop_size, max_gen=max_gen, mut_rate=mut_rate, over_fit=over_fit,
                                    log=log, plot=plot, file_lst=file_lst, repeats=repeats, overhead_fac=overhead_fac,
                                    avg_error_fac=avg_error_fac, clean_deg_len_fac=clean_deg_len_fac,
                                    clean_avg_error_fac=clean_avg_error_fac,
                                    non_unique_packets_fac=non_unique_packets_fac,
                                    unrecovered_packets_fac=unrecovered_packets_fac, chunksize=chunksize,
                                    initialize=False)
        tmp.gen_best_dist = [Distribution.Distribution.from_json(x) for x in state.get("gen_best_dist")]
        tmp.gen_avg_err = state.get("gen_avg_err")
        tmp.gen_avg_over = state.get("gen_avg_over")
        tmp.gen_calculated_error = state.get("gen_calculated_error")
        tmp.err_fit = state.get("err_fit")
        tmp.calc_err_fit = state.get("calc_err_fit")
        if tmp.calc_err_fit is not None:
            tmp.calc_err_fit = np.poly1d(tmp.calc_err_fit)
        tmp.over_fit = state.get("over_fit")
        tmp.pop = pop
        tmp.finished_gen = finished_gen
        tmp.finish_prev_best = finished_prev_best
        tmp.finished_runs_wo_imprv = finished_rungs_wo_imprv
        return tmp

    def optimize(self, start_gen=0):
        """
        Runs max_gen iterations and computes the distribution fitnesses and new generations using multiprocessing.
        :return:
        """
        prev_best = self.finish_prev_best
        runs_wo_imprv = self.finished_runs_wo_imprv
        for gen in range(start_gen, self.max_gen):
            print("########## Generation " + str(gen) + "/" + str(self.max_gen) + ". ##########")
            self.signal_handler(0, 0)
            sorted_dists = sorted(self.pop, key=lambda x: x.calculate_error_value())
            prev_best, runs_wo_imprv = self.select_best(prev_best, runs_wo_imprv, sorted_dists)
            if runs_wo_imprv >= 250:
                break
            else:
                next_gen = compute_generation(sorted_dists, self.pop_size, mut_rate=self.mut_rate)
                next_gen = self.compute_pop_fitness(next_gen)
            self.gen_best_dist.append(copy.deepcopy(sorted_dists[0]))
            self.gen_avg_err.append(sum([d.avg_err for d in self.pop]) / self.pop_size)
            self.gen_avg_over.append(sum([d.overhead for d in self.pop]) / self.pop_size)
            self.gen_clean_avg_err.append(sum([d.clean_avg_error for d in self.pop]) / self.pop_size)
            self.gen_calculated_error.append(sum([d.calculate_error_value() for d in self.pop]) / self.pop_size)
            if self.plot and gen % 25 == 0 and gen != 0:
                plot_errs_final(self.gen_best_dist, True, name=self.file + "_ev_best_results_" + str(gen))
                # plot_errs_final(self.gen_best_dist, save=True, name=self.file + "best_results_" + str(gen))
                plot_avg_errs_final(self.gen_clean_avg_err, self.gen_calculated_error, save=True,
                                    name=self.file + "clean_average_results_" + str(gen))
                plot_avg_errs_final(self.gen_avg_err, self.gen_calculated_error, save=True,
                                    name=self.file + "_ev_average_results_" + str(gen))
                save_to_csv(self.file_con, self.file + "_ev_optimization_log_" + str(gen))
            self.finished_gen = gen
            self.finished_runs_wo_imprv = runs_wo_imprv
            self.finish_prev_best = prev_best

            if self.log:
                gen_list = [gen]
                for d in self.pop:
                    gen_list.append((d.avg_err, d.overhead, d.dist_lst))
                self.file_con.append(gen_list)
            self.pop = self.create_new_gen(sorted_dists, next_gen)
        self.signal_handler(0, 0)
        if self.log:
            plot_errs_final(self.gen_best_dist, save=True, name=self.file + "best_results")
            self.err_fit, self.over_fit, self.gens = plot_avg_errs_final(self.gen_avg_err, self.gen_avg_over,
                                                                         save=True,
                                                                         name=self.file + "_ev_average_results_")
            prev_best.save_to_txt(self.file + "_ev_best_dist")
            save_to_csv(self.file_con, self.file + "_ev_optimization_log")
        return prev_best

    def compute_pop_fitness(self, pop):
        """
        Method to utilize multiprocessing for the computation of every distribution of the given population.
        :param pop:
        :return:
        """
        p = multiprocessing.Pool(self.cores)
        calc_dists = list()
        for dist in pop:
            if dist.overhead is None or dist.degree_errs is None:
                calc_dists.append(dist)
        calc_dists = p.map(self.compute_dist_fitness, calc_dists)
        p.close()
        return calc_dists

    def create_new_gen(self, pop_1, pop_2):
        """
        Merges two generations to create a new one with pop_size distributions.
        :param pop_1:
        :param pop_2:
        :return:
        """
        pop_1.extend(pop_2)
        sorted_dists = sorted(pop_1, key=lambda x: x.calculate_error_value())
        return sorted_dists[:self.pop_size]

    def select_best(self, prev_best: Distribution.Distribution, runs_wo_imprv, selected_dists):
        """
        Selects the best distribution of the current population and plots it, if it's better than the previous best.
        Adds 1 to runs_wo_imprv if no improvement were made.
        :param prev_best:
        :param runs_wo_imprv:
        :param selected_dists:
        :return:
        """
        if prev_best is not None:
            if prev_best.calculate_error_value() > selected_dists[0].calculate_error_value():
                prev_best = copy.deepcopy(selected_dists[0])
                if self.plot:
                    plot_dist(prev_best, True,
                              name=self.file + "ev_best_results_select_best_" + str(self.finished_gen))
                runs_wo_imprv = 0
            else:
                print("----- Optimal distribution has not changed. -----")
                runs_wo_imprv += 1
            print("Generations best synthetic error value: " + str(
                round(selected_dists[0].calculate_error_value(), 4)))
        else:
            prev_best = copy.deepcopy(selected_dists[0])
            if self.plot:
                plot_dist(prev_best, True,
                          name=self.file + "_ev_best_results_select_best_" + str(self.finished_gen))
        return prev_best, runs_wo_imprv
