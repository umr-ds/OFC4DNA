import json
import os
from shutil import copyfile

from Helper import *
import numpy as np
import copy

from plot import plot_avg_errs_final, plot_errs_final, plot_dist


class GradientOptimizer:
    def __init__(self, runs, alpha=None, file_lst=None, dist=None, repeats=15, overhead_fac=1.0, log=True,
                 chunksize=50, folder_append="", avg_error_fac=0.4, clean_deg_len_fac=-0.0001, clean_avg_error_fac=0.2,
                 non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, seed_spacing=0, use_payload_xor=False):
        self.calculated_error = []
        self.clean_avg_err = []
        self.avg_over = []
        self.avg_err = []
        self.gens = 0
        self.runs = runs
        self.file_lst = file_lst
        self.dist = dist
        self.dist_hist = [dist]
        self.dist_list = []
        self.repeats = repeats
        self.overhead_fac = overhead_fac
        self.log = log
        self.chunksize = chunksize
        self.avg_error_fac = avg_error_fac
        self.clean_deg_len_fac = clean_deg_len_fac
        self.clean_avg_error_fac = clean_avg_error_fac
        self.non_unique_packets_fac = non_unique_packets_fac
        self.unrecovered_packets_fac = unrecovered_packets_fac
        self.seed_spacing = seed_spacing
        self.use_payload_xor = use_payload_xor

        self.alpha, self.y = self.initialize()
        if alpha is not None:
            self.alpha = alpha
        if file_lst is None:
            self.file_lst = ['Dorn']
        try:
            if log:
                self.dir = "GrdOpt_" + str(runs) + "_" + str(repeats) + "_" + str(self.alpha) + folder_append
                self.file = self.dir + "/"
                self.file_con = list()
                os.mkdir(self.dir)
        except FileExistsError:
            print("Dir already exists. Using it anyway.")
        self.minimum_err = None
        self.err_fit = None
        self.over_fit = None

    def optimize(self):
        """
        Does the gradient descent optimization. No multiprocessing is used here since only one computation is running
        per generation. Returns the best distribution.
        :return:
        """
        self.dist.compute_fitness(self.file_lst, self.repeats, chunksize=self.chunksize)
        prev_best = self.dist
        for i in range(self.runs):
            self.gens = i
            print("########## Generation " + str(i) + "/" + str(self.runs) + ". ##########")
            self.signal_handler(0, 0)
            self.y = self.dist.calculate_error_value() - self.dist.avg_err
            theta = self.alpha * (self.runs - i) * (
                    self.y - (1.0 - (self.runs - i) / self.runs))
            self.dist = Distribution.Distribution(norm_list(shift_to_positive(self.dist.dist_lst - np.array(theta))),
                                                  overhead_fac=self.dist.overhead_fac,
                                                  avg_error_fac=self.dist.avg_error_fac,
                                                  clean_avg_error_fac=self.dist.clean_avg_error_fac,
                                                  clean_deg_len_fac=self.dist.clean_deg_len_fac,
                                                  seed_spacing=self.dist.seed_spacing,
                                                  use_payload_xor=self.dist.use_payload_xor)
            self.dist.compute_fitness(self.file_lst, self.repeats, chunksize=self.chunksize)
            if self.minimum_err is None:
                self.minimum_err = self.dist.calculate_error_value()
                plot_dist(self.dist)
            elif self.dist.calculate_error_value() < self.minimum_err:
                self.minimum_err = self.dist.calculate_error_value()
                plot_dist(self.dist)
                prev_best = copy.deepcopy(self.dist)
            self.dist_list.append(copy.deepcopy(self.dist))
            d = self.dist
            self.file_con.append(
                (d.avg_err, d.overhead, d.dist_lst, d.non_unique_packets, d.degree_errs, d.clean_deg_len,
                 d.clean_avg_error, theta))
            if self.log and i % 25 == 0:
                prev_best.save_to_txt(self.file + "best_dist")
                gen_err = [d.avg_err for d in self.dist_list]
                gen_over = [d.overhead for d in self.dist_list]
                plot_avg_errs_final(gen_err, gen_over, save=True, name=self.file + "average_results")
                save_to_csv(self.file_con, self.file + "optimization_log")
        self.signal_handler(0, 0)
        if self.log:
            plot_errs_final(self.dist_list, save=True, name=self.file + "gd_res")
            prev_best.save_to_txt(self.file + "best_dist")
            gen_err = [d.avg_err for d in self.dist_list]
            gen_over = [d.overhead for d in self.dist_list]
            self.avg_err.append(self.dist.avg_err)
            self.avg_over.append(self.dist.overhead)
            self.clean_avg_err.append(self.dist.clean_avg_error)
            self.calculated_error.append(self.dist.calculate_error_value())

            self.err_fit, self.over_fit, self.gens = plot_avg_errs_final(gen_err, gen_over, save=True,
                                                                         name=self.file + "average_results")
        save_to_csv(self.file_con, self.file + "optimization_log")
        self.dist_hist.append(prev_best)
        return prev_best

    def store_state(self, state, filename):
        def dumper(obj):
            try:
                return obj.to_json()
            except:
                return obj.__dict__

        # create all missing folders in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as outfile:
            json.dump(state, outfile, default=dumper)

    def signal_handler(self, sig, frame):
        print('Storing State...')
        filename = f"{self.dir}/grd_opt_state_" + str(self.gens) + ".json"
        self.store_state(self.get_state(), filename)
        copyfile(filename, f"{self.dir}/grd_opt_state.json")
        # sys.exit(0)

    def get_state(self):
        err_fit = list(self.err_fit) if self.err_fit is not None else None
        return {"max_gen": self.runs, "log": self.log,
                "file_lst": self.file_lst, "repeats": self.repeats,
                "overhead_fac": self.overhead_fac, "avg_error_fac": self.avg_error_fac,
                "clean_deg_len_fac": self.clean_deg_len_fac, "clean_avg_error_fac": self.clean_avg_error_fac,
                "non_unique_packets_fac": self.non_unique_packets_fac,
                "unrecovered_packets_fac": self.unrecovered_packets_fac, "alpha": self.alpha,
                "chunksize": self.chunksize,
                "gen_avg_err": self.avg_err, "avg_over": self.avg_over, "err_fit": err_fit,
                "gens": self.gens,
                "finished_gen": self.dist_list, "gen_calculated_error": self.calculated_error,
                "seed_spacing": self.seed_spacing, "use_payload_xor": self.use_payload_xor}

    def initialize(self):
        """
        Initializes the gradient optimization.
        :return:
        """
        if self.dist is None:
            self.dist = init_pop(1, add_raptor=True, overhead_fac=self.overhead_fac,
                                 avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                 clean_avg_error_fac=self.clean_avg_error_fac,
                                 non_unique_packets_fac=self.non_unique_packets_fac,
                                 unrecovered_packets_fac=self.unrecovered_packets_fac, seed_spacing=self.seed_spacing,
                                 use_payload_xor=self.use_payload_xor)[0]
        alpha = np.random.uniform(0.0001, 0.25, 1)
        y = np.ones(40)
        return alpha, y
