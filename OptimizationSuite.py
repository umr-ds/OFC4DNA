import argparse
import os

from GradientOptimizer import GradientOptimizer
from EvolutionaryOptimizer import EvolutionaryOptimizer
from CustomOptimizer import CustomOptimizer
from DifferentialOptimizer import DifferentialOptimizer
from Helper import *
import matplotlib.pyplot as plt
import time

from plot import plot_comparison_results, plot_dist


class OptimizationSuite:
    def __init__(self, file_lst=None, repeats=15, log=False, plot=False, overhead_fac=0.4, avg_error_fac=0.4,
                 clean_deg_len_fac=-0.001, clean_avg_error_fac=0.2, chunksize=None, non_unique_packets_fac=0.3,
                 unrecovered_packets_fac=0.1, seed_spacing=0, use_payload_xor=False):

        self.file_lst = file_lst
        self.repeats = repeats
        if chunksize is None:
            chunksize = [50]
        self.chunksize = chunksize
        self.log = log
        self.plot = plot

        self.overhead_fac = overhead_fac
        self.avg_error_fac = avg_error_fac
        self.clean_deg_len_fac = clean_deg_len_fac
        self.clean_avg_error_fac = clean_avg_error_fac
        self.non_unique_packets_fac = non_unique_packets_fac
        self.unrecovered_packets_fac = unrecovered_packets_fac
        self.seed_spacing = seed_spacing
        self.use_payload_xor = use_payload_xor
        if self.file_lst is None:
            self.file_lst = ['Dorn']

    def evolutionary_optimization(self, max_gen=200, pop_size=40, mut_rate=0.1, folder_append=""):
        """
        Creates an evolutionary optimizer and runs the optimization. Returns the best distribution.
        :param max_gen:
        :param pop_size:
        :param mut_rate:
        :return:
        """
        filename = f"{folder_append}/evo_opt_state.json"
        if os.path.isfile(filename):
            print(f"[Evo Opt]: Found existing state in {filename}, restoring...")
            ev_opt = EvolutionaryOptimizer.load_from_state(filename)
        else:
            ev_opt = EvolutionaryOptimizer(max_gen=max_gen, pop_size=pop_size, mut_rate=mut_rate, repeats=self.repeats,
                                           file_lst=self.file_lst, log=self.log, overhead_fac=self.overhead_fac,
                                           plot=self.plot, store_state_foldername=folder_append,
                                           avg_error_fac=self.avg_error_fac,
                                           clean_deg_len_fac=self.clean_deg_len_fac,
                                           clean_avg_error_fac=self.clean_avg_error_fac,
                                           non_unique_packets_fac=self.non_unique_packets_fac,
                                           unrecovered_packets_fac=self.unrecovered_packets_fac,
                                           chunksize=self.chunksize,
                                           use_payload_xor=self.use_payload_xor,
                                           seed_spacing=self.seed_spacing)
        return ev_opt.optimize(ev_opt.finished_gen + 1)

    def compare_evolutionary_mutation(self, mut_rates, max_gen=100, pop_size=20):
        """
        Runs the evolutionary optimization with differen mutation rates to compare the results.
        :param mut_rates:
        :param max_gen:
        :param pop_size:
        :return:
        """
        err_results = list()
        over_results = list()
        gens = [x for x in range(1, max_gen + 1)]
        for mut_rate in mut_rates:
            print(f"Running EvoOpt hyperparameter comparison with mutation rate: {mut_rate}")
            ev_opt = EvolutionaryOptimizer(max_gen=max_gen, pop_size=pop_size, mut_rate=mut_rate, repeats=self.repeats,
                                           file_lst=self.file_lst, log=self.log, plot=self.plot,
                                           store_state_foldername=f"evo_cmp_mut_{mut_rate}")
            ev_opt.optimize()
            err_results.append(ev_opt.err_fit)
            over_results.append(ev_opt.over_fit)
            del ev_opt
        plot_comparison_results(err_results, over_results, gens, mut_rates, 'M', "comp_ev")

    def compare_evolutionary_population(self, pop_sizes, max_gen=100, mut_rate=0.05):
        """
        Runs the evolutionary optimization with different population sizes to compare the results.
        :param pop_sizes:
        :param max_gen:
        :param mut_rate:
        :return:
        """
        err_results = list()
        over_results = list()
        gens = [x for x in range(1, max_gen + 1)]
        for pop_size in pop_sizes:
            print(f"Running EvoOpt hyperparameter comparison with pop_size: {pop_size}")
            ev_opt = EvolutionaryOptimizer(max_gen=max_gen, pop_size=pop_size, mut_rate=mut_rate, repeats=self.repeats,
                                           file_lst=self.file_lst, log=self.log, plot=self.plot,
                                           store_state_foldername=f"evo_cmp_pop_{pop_size}")
            ev_opt.optimize()
            err_results.append(ev_opt.err_fit)
            over_results.append(ev_opt.over_fit)
            del ev_opt
        plot_comparison_results(err_results, over_results, gens, pop_sizes, 'P', "comp_ev")

    def gradient_optimization(self, alpha=0.001, runs=200, dist=None, folder_append=""):
        """
        Creates a gradient optimizer and runs the optimization. Returns the best distribution.
        :param runs:
        :param dist:
        :return:
        """
        grd_opt = GradientOptimizer(alpha=alpha, runs=runs, repeats=self.repeats, dist=dist, file_lst=self.file_lst,
                                    log=self.log, overhead_fac=self.overhead_fac, folder_append=folder_append,
                                    avg_error_fac=avg_error_fac, clean_deg_len_fac=clean_deg_len_fac,
                                    clean_avg_error_fac=clean_avg_error_fac,
                                    non_unique_packets_fac=non_unique_packets_fac,
                                    chunksize=self.chunksize,
                                    unrecovered_packets_fac=unrecovered_packets_fac, seed_spacing=seed_spacing,
                                    use_payload_xor=use_payload_xor)
        return grd_opt.optimize()

    def compare_gradient_alpha(self, alphas, runs=250, dist=None, folder_append=""):
        """
        Runs the gradient descent optimization with different alphas to compare the results.
        :param alphas:
        :param runs:
        :param dist:
        :return:
        """
        err_results = list()
        over_results = list()
        gens = [x for x in range(1, runs + 1)]
        for alpha in alphas:
            print(f"Running GrdOpt hyperparameter comparison with alpha: {alpha}")
            grd_opt = GradientOptimizer(alpha=alpha, runs=runs, repeats=self.repeats, dist=dist, file_lst=self.file_lst,
                                        log=self.log, overhead_fac=self.overhead_fac,
                                        folder_append=f"{folder_append}_alpha_{alpha}",
                                        avg_error_fac=avg_error_fac, clean_deg_len_fac=clean_deg_len_fac,
                                        clean_avg_error_fac=clean_avg_error_fac,
                                        non_unique_packets_fac=non_unique_packets_fac,
                                        unrecovered_packets_fac=unrecovered_packets_fac, seed_spacing=seed_spacing,
                                        use_payload_xor=use_payload_xor)
            grd_opt.optimize()
            err_results.append(grd_opt.err_fit)
            over_results.append(grd_opt.over_fit)
            del grd_opt
        plot_comparison_results(err_results, over_results, gens, alphas, '\u03B1', "comp_gd")

    def differential_optimization(self, max_gen=200, pop_size=40, cr=0.8, f=1.0, folder_append=""):
        """
        Creates a differential optimizer and runs the optimization. Returns the best distribution.
        :param max_gen:
        :param pop_size:
        :param cr:
        :param f:
        :return:
        """
        filename = f"{folder_append}/diff_opt_state.json"
        if os.path.isfile(filename):
            print(f"[Diff Opt]: Found existing state in {filename}, restoring...")
            dff_opt = DifferentialOptimizer.load_from_state(filename)
        else:
            dff_opt = DifferentialOptimizer(max_gen=max_gen, pop_size=pop_size, cr=cr, f=f, file_lst=self.file_lst,
                                            log=self.log, plot=self.plot, repeats=self.repeats,
                                            overhead_fac=self.overhead_fac, avg_error_fac=self.avg_error_fac,
                                            clean_deg_len_fac=self.clean_deg_len_fac,
                                            clean_avg_error_fac=self.clean_avg_error_fac,
                                            non_unique_packets_fac=self.non_unique_packets_fac,
                                            unrecovered_packets_fac=self.unrecovered_packets_fac,
                                            chunksize=self.chunksize,
                                            store_state_foldername=filename, use_payload_xor=self.use_payload_xor,
                                            seed_spacing=self.seed_spacing)
        return dff_opt.optimize(dff_opt.finished_gen + 1)

    def compare_differential_cr(self, crs, f=0.8, max_gen=150, pop_size=40):
        """
        Runs the differential optimization with different cr to compare the results.
        :param crs:
        :param f:
        :param max_gen:
        :param pop_size:
        :return:
        """
        err_results = list()
        calc_err_results = list()
        gens = [x for x in range(1, max_gen + 1)]
        for cr in crs:
            print(f"Running DiffOpt hyperparameter comparison with cr: {cr}")
            dff_opt = DifferentialOptimizer(max_gen=max_gen, pop_size=pop_size, cr=cr, f=f, file_lst=self.file_lst,
                                            log=self.log, plot=self.plot, repeats=self.repeats,
                                            overhead_fac=self.overhead_fac, avg_error_fac=self.avg_error_fac,
                                            clean_deg_len_fac=self.clean_deg_len_fac,
                                            clean_avg_error_fac=self.clean_avg_error_fac,
                                            non_unique_packets_fac=self.non_unique_packets_fac,
                                            store_state_foldername=f"cmp_diff_opt_{cr}")
            dff_opt.optimize()
            err_results.append(dff_opt.err_fit)
            calc_err_results.append(dff_opt.calc_err_fit)
            del dff_opt
        plot_comparison_results(err_results, calc_err_results, gens, crs, 'CR', "results/comp_dff")

    def compare_differential_f(self, fs, cr=0.9, max_gen=150, pop_size=40):
        """
        Runs the differential optmization with different f to compare the results.
        :param fs:
        :param cr:
        :param max_gen:
        :param pop_size:
        :return:
        """
        err_results = list()
        over_results = list()
        gens = [x for x in range(1, max_gen + 1)]
        for f in fs:
            print(f"Running DiffOpt hyperparameter comparison with f: {f}")
            dff_opt = DifferentialOptimizer(max_gen=max_gen, pop_size=pop_size, cr=cr, f=f, file_lst=self.file_lst,
                                            log=self.log, plot=self.plot, repeats=self.repeats,
                                            overhead_fac=self.overhead_fac, avg_error_fac=self.avg_error_fac,
                                            clean_deg_len_fac=self.clean_deg_len_fac,
                                            clean_avg_error_fac=self.clean_avg_error_fac,
                                            non_unique_packets_fac=self.non_unique_packets_fac,
                                            store_state_foldername=f"diff_cmp_f_{f}")
            dff_opt.optimize()
            err_results.append(dff_opt.err_fit)
            over_results.append(dff_opt.calc_err_fit)
            del dff_opt
        plot_comparison_results(err_results, over_results, gens, fs, 'F', "results/comp_dff")

    def compare_differential_pop(self, pop_sizes, f=0.8, cr=0.9, max_gen=150):
        """
        Runs the differential optimization with different population sizes to compare the results.
        :param pop_sizes:
        :param f:
        :param cr:
        :param max_gen:
        :return:
        """
        err_results = list()
        over_results = list()
        gens = [x for x in range(1, max_gen + 1)]
        for pop_size in pop_sizes:
            print(f"Running DiffOpt hyperparameter comparison with pop: {pop_size}")
            dff_opt = DifferentialOptimizer(max_gen=max_gen, pop_size=pop_size, cr=cr, f=f, file_lst=self.file_lst,
                                            log=self.log, plot=self.plot, repeats=self.repeats,
                                            overhead_fac=self.overhead_fac, avg_error_fac=self.avg_error_fac,
                                            clean_deg_len_fac=self.clean_deg_len_fac,
                                            store_state_foldername=f"diff_cmp_pop_{pop_size}",
                                            clean_avg_error_fac=self.clean_avg_error_fac,
                                            non_unique_packets_fac=self.non_unique_packets_fac)
            dff_opt.optimize()
            err_results.append(dff_opt.err_fit)
            over_results.append(dff_opt.calc_err_fit)
            del dff_opt
        plot_comparison_results(err_results, over_results, gens, pop_sizes, 'P', "results/comp_dff")

    def custom_optimization(self, max_grades=8):
        """
        Creates a custom optimizer and runs the optimization. Returns the best distribution.
        :param max_grades:
        :return:
        """
        cst_opt = CustomOptimizer(max_grades, file_lst=self.file_lst, repeats=self.repeats, log=self.log,
                                  overhead_fac=self.overhead_fac, plot=self.plot, chunksize=self.chunksize)
        return cst_opt.optimize()

    def evo_to_grd_optimization(self):
        """
        Does the evolutionary optimization and uses the best resulting distribution for a following gradient
        optimization.
        :return:
        """
        evo_best = self.evolutionary_optimization()
        plot_dist(evo_best)
        grd_best = self.gradient_optimization(dist=evo_best)
        return grd_best

    def cst_to_grd_optimization(self):
        """
        Does the custom optimization and uses the resulting distribution for a following gradient optimization.
        :return:
        """
        cst_best = self.custom_optimization()
        plot_dist(cst_best)
        grd_best = self.gradient_optimization(dist=cst_best)
        plot_dist(grd_best)
        return grd_best

    def dff_to_grd_optimization(self):
        """
        Does the differential evolution and uses the resulting distribution for a following gradient optimization.
        :return:
        """
        dff_best = self.differential_optimization()
        plot_dist(dff_best)
        grd_best = self.gradient_optimization(dist=dff_best, runs=1000)
        plot_dist(grd_best)
        return grd_best

    def compare_repeats_raptor(self, max_repeats=100):
        """
        Compare fitness computation with different numbers of repeats to find the optimum.
        :return:
        """
        dist = Distribution.Distribution(norm_list(to_dist_list(raptor_dist)), overhead_fac=self.overhead_fac,
                                         avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                         clean_avg_error_fac=self.clean_avg_error_fac,
                                         non_unique_packets_fac=self.non_unique_packets_fac,
                                         seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        errs = list()
        over = list()
        for i in range(1, max_repeats):
            dist.compute_fitness(['Dorn'], i, chunksize=self.chunksize)
            errs.append(dist.avg_err)
            over.append(dist.overhead)
        err = plt.plot(errs, 'k', label='Average Error')
        plt.ylabel('Average Error')
        plt.xlabel('Repeats')
        plt.grid()
        plt.twinx()
        over = plt.plot(over, 'b', label='Overhead')
        plt.ylabel('Overhead')
        lns = err + over
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='best')
        plt.savefig("comp_repeats.svg", format="svg", dpi=1200)
        plt.show()

    def compute_raptor_fitness(self):
        """
        Computes the fitness of the default raptor distribution with the given parameters.
        :return:
        """
        dist = Distribution.Distribution(norm_list(to_dist_list(raptor_dist)), overhead_fac=self.overhead_fac,
                                         avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                         clean_avg_error_fac=self.clean_avg_error_fac,
                                         non_unique_packets_fac=self.non_unique_packets_fac,
                                         seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        _ = dist.compute_fitness(file_lst=self.file_lst, repeats=self.repeats, chunksize=self.chunksize)
        dist.calculate_error_value()
        plot_dist(dist, save=True)
        return dist

    def compare_chunk_size_raptor(self, chunk_sizes, file=None):
        """
        Compares the calculated average error and overhead for the raptor distribution with different chunk_sizes and
        plots the results.
        :param chunk_sizes:
        :return:
        """
        dist = Distribution.Distribution(norm_list(to_dist_list(raptor_dist)), overhead_fac=self.overhead_fac,
                                         avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                         clean_avg_error_fac=self.clean_avg_error_fac,
                                         non_unique_packets_fac=self.non_unique_packets_fac,
                                         seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        average_errs = list()
        overheads = list()
        times = list()
        err_name = "comp_chunksize_err"
        time_name = "comp_chunksize_time"
        if file is None:
            file = self.file_lst
        else:
            err_name += file
            time_name += file
            file = [file]
        for chunk_size in chunk_sizes:
            start = time.time()
            dist.compute_fitness(file_lst=file, repeats=self.repeats, chunksize=chunk_size)
            times.append((time.time() - start) / self.repeats)
            average_errs.append(dist.avg_err)
            overheads.append(dist.overhead)
        err = plt.plot(chunk_sizes, average_errs, 'ko', label='Average Error', markersize=3)
        plt.ylabel("Average Error")
        plt.xlabel("Chunk size (bytes)")
        plt.grid()
        plt.twinx()
        ove = plt.plot(chunk_sizes, overheads, 'bo', label='Overhead', markersize=3)
        plt.ylabel("Overhead")
        lns = err + ove
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='best')
        plt.savefig(err_name + ".svg", format="svg", dpi=1200)
        plt.show()
        plt.plot(chunk_sizes, times)
        plt.ylabel("Calculation time [s/Repeat]")
        plt.xlabel("Chunks size (bytes)")
        plt.savefig(time_name + ".svg", format='svg', dpi=1200)
        plt.show()

    def compute_fitness_from_input(self):
        """
        Takes a list-formatted list as input to create a distribution from it and calculates the fitness of the
        distribution.
        :return:
        """
        dist_str = input("Enter a distribution list: ")
        dist_str = dist_str.replace("[", "").replace("]", "").replace(",", "")
        dist_lst = dist_str.split()
        dist_lst = [float(val) for val in dist_lst]
        dist = Distribution.Distribution(norm_list(dist_lst), overhead_fac=self.overhead_fac,
                                         avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                         clean_avg_error_fac=self.clean_avg_error_fac,
                                         non_unique_packets_fac=self.non_unique_packets_fac,
                                         seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        dist.compute_fitness(file_lst=self.file_lst, repeats=self.repeats, chunksize=self.chunksize)
        plot_dist(dist, dist.overhead_fac)

    def compute_fitness_from_txt(self, filename):
        """
        Reads a distribution from a file and computes the fitness.
        :param filename:
        :return:
        """
        dist = self.read_from_txt(filename)
        dist.compute_fitness(file_lst=self.file_lst, repeats=self.repeats, chunksize=self.chunksize)
        plot_dist(dist, dist.overhead_fac)

    def read_from_txt(self, filename):
        """
        Reads a distribution from a textfile created with the save_to_txt() method of the distribution class and returns
        a distribution object.
        :param filename:
        :return:
        """
        with open(filename, 'r') as f:
            dist_str = f.readline().split("[")[1].split("]")[0].replace(",", "")
            dist_lst = dist_str.split()
            dist_lst = [float(val) for val in dist_lst]
        dist = Distribution.Distribution(norm_list(dist_lst), overhead_fac=self.overhead_fac,
                                         avg_error_fac=self.avg_error_fac, clean_deg_len_fac=self.clean_deg_len_fac,
                                         clean_avg_error_fac=self.clean_avg_error_fac,
                                         non_unique_packets_fac=self.non_unique_packets_fac,
                                         seed_spacing=self.seed_spacing, use_payload_xor=self.use_payload_xor)
        return dist


if __name__ == '__main__':
    # Example input files / classes - grouped by entropy
    bmp_low_entropy = ["logo_mosla_bw.bmp"]
    image_high_entropy = ["logo.jpg", "logo_mosla_rgb.png", "Marburg_0192_Elektroll_CC0.png"]
    compress_encrypt_high_entropy = ["Dorn.zip", "aes_Dorn", "aes_ecb_Dorn"] + image_high_entropy
    text_medium_entropy = ['Dorn', 'LICENSE', 'Rapunzel', 'Rothkäppchen', 'Sneewittchen']
    text_medium_high_entropy = ["Dorn.pdf", "lorem_ipsum100k.doc"]

    parser = argparse.ArgumentParser(description="Script to set various parameters")

    parser.add_argument("--files", nargs="+", default=["aes_Dorn"], help="One or more files to encode and optimize for")
    # Error-function Parameters (optimization goals)
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats")
    parser.add_argument("--chunksize_lst", type=int, nargs="+", default=[40, 60, 80], help="List of chunk sizes")
    parser.add_argument("--overhead_fac", type=float, default=0.2, help="Overhead factor")
    parser.add_argument("--avg_error_fac", type=float, default=0.4, help="Average error factor")
    parser.add_argument("--clean_deg_len_fac", type=float, default=-0.0001, help="Clean degree length factor")
    parser.add_argument("--clean_avg_error_fac", type=float, default=0.2, help="Clean average error factor")
    parser.add_argument("--non_unique_packets_fac", type=float, default=0.3, help="Non-unique packets factor")
    parser.add_argument("--unrecovered_packets_fac", type=float, default=0.1, help="Unrecovered packets factor")

    # Hyper-Parameter
    parser.add_argument("--gens", type=int, default=250, help="Number of generations")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size (only for diff and evo)")
    # diff
    parser.add_argument("--cr", type=float, default=0.8, help="Crossover rate (for diff)")
    parser.add_argument("--f", type=float, default=0.8, help="Scaling factor (for diff)")
    # evo
    parser.add_argument("--mut_rate", type=float, default=0.2, help="Mutation rate (for evo)")
    # grd
    parser.add_argument("--alpha", type=float, default=0.001, help="Alpha (for grd)")

    # Raptor modifiers
    parser.add_argument("--seed_spacing", type=int, default=4, help="Seed spacing (0 is same as no seed-spacing)")
    parser.add_argument("--use_payload_xor", type=bool, default=True, help="Use payload XOR (True/False)")

    # Hyper-Parameter Comparison
    parser.add_argument("--compare_pop_size", type=int, default=100,
                        help="Population size for hyperparameter comparison")
    parser.add_argument("--compare_gens", type=int, default=100,
                        help="Number of generations for hyperparameter comparison")

    args = parser.parse_args()

    # Access the variables
    repeats = args.repeats
    chunksize_lst = args.chunksize_lst
    overhead_fac = args.overhead_fac
    avg_error_fac = args.avg_error_fac
    clean_deg_len_fac = args.clean_deg_len_fac
    clean_avg_error_fac = args.clean_avg_error_fac
    non_unique_packets_fac = args.non_unique_packets_fac
    unrecovered_packets_fac = args.unrecovered_packets_fac

    gens = args.gens
    pop_size = args.pop_size
    cr = args.cr
    f = args.f
    mut_rate = args.mut_rate
    alpha = args.alpha

    seed_spacing = args.seed_spacing
    use_payload_xor = args.use_payload_xor

    compare_pop_size = args.compare_pop_size
    compare_gens = args.compare_gens

    # This code can also be used without argparse:
    """
    # Error-function Parameters (optimization goals):
    repeats = 10
    chunksize_lst = [40, 60, 80]
    overhead_fac = 0.2
    avg_error_fac = 0.4
    clean_deg_len_fac = -0.0001
    clean_avg_error_fac = 0.2
    non_unique_packets_fac = 0.3
    unrecovered_packets_fac = 0.1

    # Hyper-Parameter:
    gens = 250
    pop_size = 100  # (only diff and evo)
    # diff:
    cr = 0.8
    f = 0.8
    # evo:
    mut_rate = 0.2
    # grd:
    alpha = 0.001

    # Raptor modifiers (increases input entropy)
    seed_spacing = 4  # 0 is same as no seed-spacing
    use_payload_xor = True

    # Hyper-Parameter Comparison:
    compare_pop_size = 100  # size of the population for hyperparameter comparison
    compare_gens = 100  # number of generations for hyperparameter comparison
    """

    suite = OptimizationSuite(repeats=repeats, plot=True,
                              file_lst=bmp_low_entropy,
                              # or manual: ['Dorn', 'LICENSE', 'Rapunzel', 'Rothkäppchen', 'Shneewittchen'],
                              chunksize=chunksize_lst, log=True, overhead_fac=overhead_fac, avg_error_fac=avg_error_fac,
                              clean_deg_len_fac=clean_deg_len_fac, clean_avg_error_fac=clean_avg_error_fac,
                              non_unique_packets_fac=non_unique_packets_fac,
                              unrecovered_packets_fac=unrecovered_packets_fac, seed_spacing=seed_spacing,
                              use_payload_xor=use_payload_xor)

    # Optimization (DE)
    print("Differential Evolution")
    diff_best = suite.differential_optimization(max_gen=gens, pop_size=pop_size, cr=cr, f=f,
                                                folder_append="results_diff")
    # Optimization (EA)
    print("Evolutionary Algorithm")
    evo_best = suite.evolutionary_optimization(max_gen=gens, pop_size=pop_size, mut_rate=mut_rate,
                                               folder_append="results_evo")
    # Optimization (GD)
    print("Gradient Descent")
    suite.gradient_optimization(alpha=alpha, runs=gens, folder_append="results_grd",
                                dist=Distribution.Distribution(norm_list(to_dist_list(raptor_dist)), overhead_fac,
                                                               avg_error_fac, clean_deg_len_fac, clean_avg_error_fac,
                                                               non_unique_packets_fac, unrecovered_packets_fac,
                                                               seed_spacing=seed_spacing,
                                                               use_payload_xor=use_payload_xor))

    print("Compare Differential Evolution with different F values")
    suite.compare_differential_f(fs=[0.7, 0.8, 0.9, 1.0], cr=0.8, max_gen=compare_gens, pop_size=compare_pop_size)
    print("Compare Differential Evolution with different CR values")
    suite.compare_differential_cr(crs=[0.1, 0.2, 0.4, 0.6, 0.8], f=0.8, max_gen=compare_gens)
    print("Compare Evolutionary Algorithm with different mutation rates")
    suite.compare_evolutionary_mutation(mut_rates=[0.1, 0.2, 0.3, 0.4, 0.5], max_gen=compare_gens,
                                        pop_size=compare_pop_size)
    print("Compare Evolutionary Algorithm with different population sizes")
    suite.compare_evolutionary_population(pop_sizes=[25, 50, 75, 100, 125, 150], max_gen=compare_gens, mut_rate=0.2)
    print("Compare Gradient Descent with different alpha values")
    suite.compare_gradient_alpha(alphas=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], runs=compare_gens)

    # fine-tune optimized distributions using gradient descent:
    diff_to_grd_best = suite.gradient_optimization(dist=diff_best, runs=compare_gens, folder_append="_diff_to_grd")
    plot_dist(diff_to_grd_best)

    evo_to_grd_best = suite.gradient_optimization(dist=evo_best, runs=compare_gens, folder_append="_evo_to_grd")
    plot_dist(evo_to_grd_best)
