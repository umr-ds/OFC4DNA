import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import Helper
from Distribution import Distribution

sns.set(font_scale=1.25)
sns.set_style("ticks", {'axes.grid': True})
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
matplotlib.rcParams.update({'figure.autolayout': True}),


def plot_dist(dst, save=False, name=None):
    """
    Simple plot of the distribution.
    :dist: distribution to plot.
    :param save: If true, the plot will be saved as svg file.
    :param name: Filename to save the plot with.
    :return:
    """
    dist = plt.plot([x for x in range(0, 41)], [x * 100 for x in [0] + dst.dist_lst], 'b',
                    label='distribution function')
    plt.xlim([1, 40])
    plt.xlabel("degree")
    plt.ylabel("probability [%]")
    err = plt.plot([], [], ' ', label="avg error: " + str(round(dst.avg_err, 5)))
    over = plt.plot([], [], ' ', label="overhead: " + str(round(dst.overhead, 3)))
    clean_avg_error = plt.plot([], [], ' ', label="clean avg err: " + str(round(dst.clean_avg_error, 3)))
    clean_deg_len = plt.plot([], [], ' ', label="clean deg len: " + str(round(dst.clean_deg_len, 3)))
    calculated_error_value = plt.plot([], [], ' ', label="syn. error: " + str(round(dst.calculated_error_value, 5)))
    non_unique_packets = plt.plot([], [], ' ', label="non unique packets: " + str(round(dst.non_unique_packets, 3)))
    lns = dist + err + over + clean_avg_error + clean_deg_len + calculated_error_value + non_unique_packets
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='best')
    plt.grid()
    os.makedirs("results", exist_ok=True)
    if save:
        if name is None:
            name = "results/dist" + str(round(dst.avg_err, 2))
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    plt.close('all')


def plot_dist_with_raptor(dst, save=False, name=None):
    """
    Simple plot of the distribution.
    :dist: distribution to plot.
    :param save: If true, the plot will be saved as svg file.
    :param name: Filename to save the plot with.
    :return:
    """
    rap_list = Helper.norm_list(Helper.to_dist_list(Helper.raptor_dist))
    rap = plt.plot([x for x in range(1, 41)], [x * 100 for x in rap_list], 'r',
                   label='Raptor distribution function')
    dist = plt.plot([x for x in range(1, 41)], [x * 100 for x in dst.dist_lst], 'b',
                    label='distribution function EA')
    plt.xlabel("degree")
    plt.ylabel("probability [%]")
    lns = dist + rap
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='best')
    plt.grid()
    if save:
        if name is None:
            name = "dist" + str(round(dst.calculated_error_value, 3))
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    plt.close('all')


def plot_errs_final(dists, save=False, name=None):
    """
    Plots the errors of all the generations best distributions generated while optimizing.
    :param dists: The best distributions of every generation.
    :param save: If true, saves the file as .svg.
    :param name: The name to save the file with.
    :return:
    """
    err_list = []
    for dist in dists:
        err_list.append(dist.avg_err)
    gens = [x for x in range(1, len(err_list) + 1)]
    coef = np.polyfit(gens, err_list, 1)
    poly1d = np.poly1d(coef)
    plt.plot(gens, err_list, 'k', poly1d(gens), '--r')
    plt.xlabel('Generation')
    plt.ylabel('best average error')
    plt.grid()
    if save:
        if name is None:
            name = "opt_res"
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    plt.close('all')
    return poly1d


def plot_avg_errs_final(avg_errs, gen_calculated_error=None, save=False, name=None):
    """
    Plots the average errors (and overheads) of every generation.
    :param avg_errs: List of average errors for every generation.
    :param gen_calculated_error: List of average overheads for every generation.
    :param save: If true, saves the file as .svg.
    :param name: The name to save the file with.
    :return:
    """
    gens = [x for x in range(1, len(avg_errs) + 1)]
    err_coefs = np.polyfit(gens, avg_errs, 1)
    calc_err_coefs = np.polyfit(gens, gen_calculated_error, 1)
    err_poly1d = np.poly1d(err_coefs)
    calc_err_poly1d = np.poly1d(calc_err_coefs)  # convert to drawable function
    err = plt.plot(gens, avg_errs, 'ok', label="average error", markersize=3)
    err_f = plt.plot(gens, err_poly1d(gens), '--k', label=str(err_poly1d).split("\n")[1])
    plt.xlabel('Generation')
    plt.ylabel('average error')
    plt.grid()
    plt.twinx()
    calc_err = plt.plot(gens, gen_calculated_error, 'ob', label="average synthetic error", markersize=3)
    calc_err_f = plt.plot(gens, calc_err_poly1d(gens), '--b', label=str(calc_err_poly1d).split("\n")[1])
    plt.ylabel('average synthetic error')
    lns = err + calc_err + err_f + calc_err_f
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='best')
    if save:
        if name is None:
            name = "opt_avg_res"
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    plt.close('all')
    return err_poly1d, calc_err_poly1d, gens


def plot_from_pop_csv_log(filename, save=False, name=None, overhead_fac=0.5):
    """
    Used this to plot the results from optimization_logs
    :param filename:
    :param save:
    :param name:
    :param overhead_fac:
    :return:
    """
    err_list = list()
    dists = list()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                if row != 0:
                    tmp = 1.0
                    for i in range(1, 40):
                        err = float(row[i].split(',')[0].replace('(', ''))
                        over = float(row[i].split(',')[1])
                        cost = err + (overhead_fac * over)
                        if tmp > cost:
                            tmp = cost
                    err_list.append(tmp)
                    str_list = row[i].split('[')[1].split(']')[0].split(',')
                    dist = [float(x) for x in str_list]
                    dists.append(Distribution(dist))
                    dists[-1].avg_err = err_list[-1]
            except Exception:
                print(row)
    gens = [x for x in range(1, len(err_list) + 1)]
    coef = np.polyfit(gens, err_list, 1)
    poly1d = np.poly1d(coef)
    plt.plot(gens, err_list, 'k')
    plt.plot(gens, poly1d(gens), '--r', label=str(poly1d).split("\n")[1].split("+")[0])
    plt.xlabel('Generation')
    plt.ylabel('average error + weighted Overhead')
    plt.grid()
    plt.legend(loc='best')
    if save:
        if name is None:
            name = "opt_res"
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    plt.close('all')
    return poly1d


def plot_multiple_evo_res(gen_best_res, title):
    """
    Plots the results for the comparison functions of the evolutionary optimizer.
    :param gen_best_res:
    :param title:
    :return:
    """
    for elem in gen_best_res:
        err_list = [x.avg_err for x in elem[1]]
        gens = [x for x in range(1, len(err_list) + 1)]
        plt.plot(gens, err_list, label=elem[0])
    plt.xlabel('Generation')
    plt.ylabel('best average error per Generation')
    plt.legend(loc='best')
    plt.title(title)
    plt.show(block=False)
    plt.close('all')


def plot_comparison_results(err_res, calcualted_syn_err_val, gens, param_vals, param, method):
    """
    Method to plot the results of the 'compare' methods.
    :param err_res:
    :param calcualted_syn_err_val:
    :param gens:
    :param param_vals:
    :param param:
    :param method:
    :return:
    """
    for i in range(len(err_res)):
        plt.plot(gens, err_res[i](gens), label=param + " = " + str(param_vals[i]))
    plt.xlabel('Generation')
    plt.ylabel('average error')
    plt.ylabel('error probability')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(method + "_err_" + param + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    for i in range(len(calcualted_syn_err_val)):
        plt.plot(gens, calcualted_syn_err_val[i](gens), label=param + " = " + str(param_vals[i]))
    plt.xlabel('Generation')
    plt.ylabel('average calculated synthetic error value')
    plt.ylabel('calculated synthetic error value')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(method + "_over_" + param + ".svg", format="svg", dpi=1200)
    plt.show(block=False)
    plt.close('all')
    lines = [str(param_vals[i]) + ", " + str(err_res[i].c[0]) + ", " + str(calcualted_syn_err_val[i].c[0]) + "\n" for i
             in
             range(len(err_res))]
    with open(method + "_err_" + param + ".txt", 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    import json

    for folder in ["all_1000", "all", "bmp_low_entropy", "image_high_entropy", "text_medium_entropy",
                   "text_medium_high_entropy"]:
        fille = f"../dist_opt_results/{folder}/_data/evo_opt_state.json"

        with open(fille, "r") as ff_file:
            tmp = json.load(ff_file)
            print(tmp)
        overhead_lst = []
        avg_err_lst = []
        clean_deg_len_lst = []
        clean_avg_error_lst = []
        calculated_erro_value_lst = []

        for gen_dist in tmp["gen_best_dist"]:
            overhead_lst.append(gen_dist["overhead"])
            avg_err_lst.append(gen_dist["avg_err"])
            clean_deg_len_lst.append(gen_dist["clean_deg_len"])
            clean_avg_error_lst.append(gen_dist["clean_avg_error"])
            calculated_erro_value_lst.append(gen_dist["calculated_error_value"] / 100.0)

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax2.plot(overhead_lst, label="overhead")
        ax2.set_ylabel("Overhead")
        ax1.plot(avg_err_lst, label="average error")
        ax1.set_ylabel("Average error")
        ax1.plot(calculated_erro_value_lst, label="Calculated error value", color='red')
        plt.title(folder.replace("_", " "))
        plt.grid(True)
        ax1.legend(loc=0)
        ax2.autoscale_view()
        ax2.autoscale_view()
        ax1.set_ylim(min(calculated_erro_value_lst), max(calculated_erro_value_lst))
        plt.grid(True)
        plt.show(block=False)
        """
        plt.plot([x / 100.00 for x in tmp["gen_calculated_error"]])
        plt.plot(tmp["gen_avg_err"])
        plt.plot(tmp["gen_avg_over"])
        """
