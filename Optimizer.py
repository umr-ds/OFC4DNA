import multiprocessing
import os
import json

from Helper import init_pop


class Optimizer:
    def __init__(self, pop_size, max_gen, log=True, plot=False, file_lst=None, repeats=15,
                 overhead_fac=0.2, avg_error_fac=0.4, clean_deg_len_fac=-0.0001, clean_avg_error_fac=0.2,
                 non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, chunksize=50, initialize=True,
                 store_state_foldername=None, seed_spacing=0, use_payload_xor=False):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.log = log
        self.plot = plot
        self.file_lst = file_lst
        self.repeats = repeats
        self.overhead_fac = overhead_fac
        self.avg_error_fac = avg_error_fac
        self.clean_deg_len_fac = clean_deg_len_fac
        self.clean_avg_error_fac = clean_avg_error_fac
        self.non_unique_packets_fac = non_unique_packets_fac
        self.unrecovered_packets_fac = unrecovered_packets_fac
        self.seed_spacing = seed_spacing
        self.use_payload_xor = use_payload_xor
        if store_state_foldername is None:
            store_state_foldername = "results"
        self.store_state_foldername = store_state_foldername
        if not isinstance(chunksize, list):
            self.chunksize = [chunksize]
        else:
            self.chunksize = chunksize
        if self.file_lst is None:
            self.file_lst = ['Dorn']
        self.pop = None
        self.cores = multiprocessing.cpu_count() - 1
        if self.cores > self.pop_size:
            self.cores = self.pop_size

        self.gen_best_dist = list()
        self.gen_avg_err = list()
        self.gen_avg_over = list()
        self.gen_clean_avg_err = list()
        self.gen_calculated_error = list()
        self.err_fit = None
        self.calc_err_fit = None
        self.gens = None

        self.finished_gen = -1
        self.finish_prev_best = None
        self.finished_runs_wo_imprv = 0

        if initialize:
            self.initialize()

    def optimize(self):
        """
        not implemented in base class
        """
        pass

    def create_dist_from_res(self):
        """
        not implemented in base class
        """
        pass

    def initialize(self, add_raptor=True):
        """
        Initializes the start population with the raptor distribution and randomized distributions and evaluates the
        fitness for every distribution.
        :param add_raptor:
        :return:
        , add_raptor=False, overhead_fac=0.2, avg_error_fac=0.4, clean_deg_len_fac=-0.0001,
             clean_avg_error_fac=0.2, non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, seed_spacing=0,
             use_payload_xor=False
        """
        pop = init_pop(self.pop_size, add_raptor=add_raptor, overhead_fac=self.overhead_fac,
                       avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                       clean_avg_error_fac=self.clean_avg_error_fac, non_unique_packets_fac=self.non_unique_packets_fac,
                       unrecovered_packets_fac=self.unrecovered_packets_fac, seed_spacing=self.seed_spacing,
                       use_payload_xor=self.use_payload_xor)
        p = multiprocessing.Pool(self.cores)
        try:
            self.pop = p.map(self.compute_dist_fitness, pop)  # [self.compute_dist_fitness(x) for x in pop]
        except Exception as _:
            # fallback for better debug...
            self.pop = [self.compute_dist_fitness(x) for x in pop]
        p.close()

    def compute_dist_fitness(self, dist):
        """
        Help method to make multiprocessing easier. Calls the compute_fitness method for a given distribution and
        returns the distribution.
        :param dist: Distribution to compute the fitness for.
        :return:
        """
        dist.compute_fitness(self.file_lst, repeats=self.repeats, chunksize=self.chunksize)
        return dist

    def get_state(self):
        err_fit = list(self.err_fit) if self.err_fit is not None else None
        return {"pop_size": self.pop_size, "max_gen": self.max_gen, "log": self.log,
                "plot": self.plot, "file_lst": self.file_lst, "repeats": self.repeats,
                "overhead_fac": self.overhead_fac, "avg_error_fac": self.avg_error_fac,
                "clean_deg_len_fac": self.clean_deg_len_fac, "clean_avg_error_fac": self.clean_avg_error_fac,
                "non_unique_packets_fac": self.non_unique_packets_fac,
                "unrecovered_packets_fac": self.unrecovered_packets_fac,
                "chunksize": self.chunksize, "pop": self.pop, "gen_best_dist": self.gen_best_dist,
                "gen_avg_err": self.gen_avg_err, "gen_avg_over": self.gen_avg_over, "err_fit": err_fit,
                "calc_err_fit": None if self.calc_err_fit is None else list(self.calc_err_fit), "gens": self.gens,
                "finished_gen": self.finished_gen, "gen_calculated_error": self.gen_calculated_error,
                "finished_prev_best": self.finish_prev_best, "finished_runs_wo_imprv": self.finished_runs_wo_imprv,
                "seed_spacing": self.seed_spacing, "use_payload_xor": self.use_payload_xor}

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

    def compute_pop_fitness(self, pop):
        """
        not implemented in base class
        """
        pass
