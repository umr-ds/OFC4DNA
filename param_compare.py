import csv
import json
import math
import multiprocessing
import os
import zipfile
from datetime import datetime

import matplotlib

# matplotlib.use('pdf')
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Distribution
from Helper import encode, calculate_entropy, norm_list, to_dist_list, raptor_dist, encode_for_spacing
from NOREC4DNA.norec4dna.distributions.RaptorDistribution import RaptorDistribution
from NOREC4DNA.norec4dna.rules.FastDNARules import FastDNARules

# increase seaborn font size
sns.set(font_scale=1.25)
# matplotlib.rcParams.update({'figure.autolayout': True})

files = ["Dorn", "logo.jpg", "logo_mosla_bw.bmp", "Uni_Marburg_Logo.bmp", "Uni_Marburg_Siegel_sw.bmp", "Dorn.zip"]
chunk_sizes = [40, 60, 80, 100]
dists = [RaptorDistribution(100).f]
seed_struct_strs = ["H"]
rules = FastDNARules()
return_packets = False
repeats = 0
id_spacings = [0, 1, 2, 3, 4, 5, 6]


def process_combination(args):
    file, chunk_size, dist, id_spacing, use_payload_xor, mask_id, seed_struct_str = args
    packets, number_of_chunks = encode(file=file, chunk_size=chunk_size, dist=dist, rules=rules,
                                       return_packets=return_packets, repeats=repeats, id_spacing=id_spacing,
                                       mask_id=mask_id, use_payload_xor=use_payload_xor,
                                       seed_struct_str=seed_struct_str, return_packet_error_vals=True,
                                       store_packets=False)
    err_nums = np.array(packets, dtype=np.double)
    avg_error = np.average(err_nums)
    variance_error = np.var(err_nums)
    rule_violating_packets = err_nums[err_nums > 1.0].size
    # store each err_num in a csv row:
    # with open(
    #        f"err_nums_{file}_{mask_id}_{chunk_size}_{id_spacing}_{use_payload_xor}_{seed_struct_str}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
    #        "a") as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["err_nums"])
    #    writer.writerows([[x] for x in err_nums.tolist()])
    del packets
    dna_entropy = calculate_entropy(file)
    entropy = calculate_entropy(file, convert_to_dna=False)
    return [file, chunk_size, dist, id_spacing, use_payload_xor, mask_id, seed_struct_str, avg_error, variance_error,
            rule_violating_packets, number_of_chunks, entropy, dna_entropy]


def create_graphs(csv_filename):
    # load csv into pandas dataframe:
    df = pd.read_csv(csv_filename)
    # replace "_" in all column names with " ":
    df.columns = df.columns.str.replace("_", " ").str.capitalize()

    # add a column "seed_len" to the dataframe:
    df["seed_struct_str"] = "H"
    rows_of_interest = ["File", "Rule violating packets", "Id spacing", "Use payload xor", "Avg error", "Chunk size",
                        "Mask id"]
    for hue_to_use in ["File", "Id spacing", "Use payload xor", "Chunk size", "Mask id"]:
        sns.pairplot(data=df[rows_of_interest], hue=hue_to_use, palette="deep")
        plt.savefig(f"pair_{hue_to_use}.svg", format="svg", dpi=1200)
        plt.savefig(f"pair_{hue_to_use}.pdf", bbox_inches="tight")
        plt.show(block=False)
    # sns.scatterplot(data=df, x="chunk_size", y="rule_violating_packets", hue="file",
    #                style="id_spacing", size="use_payload_xor", palette="deep")
    g = df.groupby(["File", "Chunk size"])
    for file_name, group in g:
        file_name, chunk_size = file_name
        pl = sns.scatterplot(data=group, x="Id spacing", y="Rule violating packets", hue="Mask id",
                             style="Use payload xor", palette="deep")
        pl.set_ylabel("Rule violating packets")
        pl.set_title(f"{file_name} - Chunksize: {chunk_size}")
        plt.savefig(f"err_{file_name}_{chunk_size}.svg", format="svg", dpi=1200)
        plt.savefig(f"err_{file_name}_{chunk_size}.pdf", bbox_inches="tight")
        plt.show(blocK=False)

    g2 = df.groupby(["Id spacing", "Chunk size", "File"]).mean()
    sns.scatterplot(data=g2, x="Id spacing", y="Rule violating packets", hue="Chunk size", style="File", palette="deep")
    pl.set_ylabel("Rule violating packets")
    plt.savefig(f"err_scatter_{file_name}_{chunk_size}.svg", format="svg", dpi=1200)
    plt.savefig(f"err_scatter_{file_name}_{chunk_size}.pdf", bbox_inches="tight")
    plt.show(block=False)
    print(df)

    # compare rule_violating_packets vs id_spacing
    # Create the Seaborn plot
    # sns.set(style="whitegrid")
    tmp = df[(df.use_payload_xor == False) & (df.mask_id == False)].groupby("file")
    for file_name, tmp_data in tmp:
        # plt.figure(figsize=(10, 6))
        grouped_data = tmp_data.groupby(['chunk_size'])  # .agg(mean=("rule_violating_packets", "mean"))
        # grouped_data.reset_index()
        for chunk_size, group in grouped_data:
            sns.barplot(x="Id spacing", y="Rule violating packets", errorbar='sd', capsize=0.09, data=group,
                        label=chunk_size)
        # Create the Seaborn line plot with error bars
        # plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        try:
            plot = sns.lineplot(x="Id spacing", y="mean", hue="Chunk size", data=grouped_data)
            plot = pd.crosstab(tmp_data.chunk_size, tmp_data.id_spacing, tmp_data.rule_violating_packets).plot(
                kind='bar')
            plot.errorbar(grouped_data["Id spacing"], grouped_data["mean"], yerr=grouped_data["std"], fmt='o',
                          markersize=4)

            plot.set_title(f"Influence of 'id spacing' on 'rule violating packets' {file_name}")
            plot.set_xlabel("Id spacing")
            plot.set_ylabel("Mean 'rule violating packets'")
            plt.autoscale()
            plt.savefig(f"line_{file_name}.svg", format="svg", dpi=1200)
            plt.savefig(f"line_{file_name}.pdf", bbox_inches="tight")
            plt.show(block=False)
        except ValueError:
            print(f"ValueError for {file_name}")
    # graph = df.pivot_table(index="id_spacing", columns="rule_violationg_packets").plot.bar()
    ##graph = sns.barplot(data=g3, x="id_spacing", y="rule_violating_packets", palette="deep")
    # graph.set_title(f"rule_violating_packets vs id_spacing")
    # plt.savefig(f"bar_rule_violating_vs_id_spacing.svg", format="svg", dpi=1200)
    # plt.savefig(f"bar_rule_violating_vs_id_spacing.pdf", bbox_inches="tight")
    # plt.show(block=False)
    # compare rule_violating_packets vs chunk_size
    # g4 =
    # compare rule_violating_packets vs use_payload_xor
    # compare rule_violating_packets vs mask_id


def plot_err_nums(label_file_names):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "pink", "brown", "black"]
    lines = ["--", ":", "-.", ":"]
    plt.figure(figsize=(10, 6))
    for idx, label_file_name in enumerate(label_file_names):
        label, file_name = label_file_name
        df = pd.read_csv(file_name)
        # line plot of err_nums:
        # plot each line of the csv as a value for a single line plot:
        # use only the first 500 rows of the dataframe:
        df = df.head(300)
        # cap the maximum value to 1.0:
        df = df.applymap(lambda x: 1.0 if x > 1.0 else x)
        plot = sns.lineplot(x=range(df.__len__()), linestyle=lines[idx], y=df.err_nums, color=colors[idx],
                            label=label)
    plot.set_xlabel("Packet number")
    plot.set_ylabel("Calculated error probability")
    plot.set_title("Calculated error probability for the first 300 packets with a 2 byte seed")
    plt.savefig("id_to_err_nums_dist" + ".pdf", bbox_inches="tight")
    plt.savefig("id_to_err_nums_dist" + ".svg")
    plt.show(block=False)


def get_packets(seed_struct_str="H"):
    packets = []
    # for i in [True, False]:
    i = True  # False
    #"""
    # the following does only work for sleedping_beauty, chunk_size=4, default dist, id_spacing=0, , mask_id=False, use_payload_xor=False, seed_struct_str="I"
    # check if "tmp_packets.json" exits:
    if os.path.exists("tmp_packets.json") and seed_struct_str == "I":
        with open("tmp_packets.json", "r") as f:
            packets = json.load(f)
    else:
        packets.append(
            encode_for_spacing(file="sleeping_beauty", chunk_size=20, dist=dists[0], id_spacing=0, mask_id=False,
                               use_payload_xor=False, seed_struct_str=seed_struct_str))
        # dump current packets to a file:
        with open("tmp_packets.txt", "w") as f:
            json.dump(packets, f)

    for spacing in [6,7]:  # range(8):
        # packets.append(encode(file="sleeping_beauty", chunk_size=40, dist=dists[0], rules=rules,
        #                       return_packets=True, repeats=repeats, id_spacing=spacing,
        #                      mask_id=False, use_payload_xor=i,
        #                      seed_struct_str=seed_struct_str, return_packet_error_vals=True,
        #                      store_packets=False)[0])
        pkts_spacing = encode_for_spacing(file="sleeping_beauty", chunk_size=20, dist=dists[0], id_spacing=spacing,
                                          mask_id=False, use_payload_xor=i, seed_struct_str=seed_struct_str)
        with open(f"tmp_packets_{spacing}.txt", "w") as f:
            json.dump(pkts_spacing, f)
        packets.append(pkts_spacing)
    diff = [(packets[0][i], packets[1][i]) for i in range(len(packets[0]))]
    with open("diff.json", "w") as f:
        json.dump(diff, f)
    # print([(packets[0][i], packets[4][i]) for i in range(len(packets[0])) if
    #       packets[0][i].error_prob > packets[4][i].error_prob])
    data = [(len([(packets[0][i], packets[k][i]) for i in range(len(packets[0])) if
                  packets[0][i] > packets[k][i] and packets[0][i] >= 1.0]),

             len([(packets[0][i], packets[k][i]) for i in range(len(packets[0])) if
                  packets[0][i] < packets[k][i] and packets[0][i] < 1.0])) for k in
            range(len(packets))]
    """
    #matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)
    # data for 2 byte seed (laxer rules, long sequences):
    data = [(0, 0), (1118, 945), (1606, 1153), (1982, 1262), (2280, 1353), (2341, 1460), (2407, 1434), (2559, 1482)]
    # data for 2 byte seed (typically used rules):
    data = [(0, 0), (26262, 10493), (26398, 10675), (26434, 10689), (26393, 10730), (26369, 10639), (26334, 10742), (26311, 10757), (26461, 10700)]
    # data for 4 byte seed (laxer rules, long sequences):
    data = [(0, 0), (153358975, 129382698), (208309123, 165962976), (289514196, 187265992), (338349469, 198733809), (351124970, 208058394), (367586972, 210544770), (376977775, 213091391)]
    #data for 4 byte seed (typically used rules):
    data = [(0, 0), (1370968148, 607618858), (1375851326, 612796209), (1374106800, 610914708)] (0 spacing, 2 spacing, 4 spacing, 6 spacing)
    # data for 4 byte seed (max allowed homopolymer run = 2):
    data = [(0, 0), (1332159635, 28458764), (1335852259, 28446819), (1335778244, 28445299)]
    """
    print(data)
    print([x-y for x,y in data])
    print(max([x - y for x, y in data]))
    with open("tmp_data.txt", "w") as f:
        json.dump(data, f)
    # Extract the x values (index of tuples) starting from 1
    x_values = np.array([2,4,6,8]) # np.arange(1, len(data) + 1)

    # Extract the y values (both the first and second elements of each tuple)
    y_values_1 = [x[0] for x in data[1:]]  # Skip the first pair
    y_values_2 = [x[1] for x in data[1:]]  # Skip the first pair

    # Width of the bars
    bar_width = 0.35

    # Create the barplot with two bars per index
    plt.bar(x_values[:-1] - bar_width / 2, y_values_1, width=bar_width, label='valid')
    plt.bar(x_values[:-1] + bar_width / 2, y_values_2, width=bar_width, label='invalid')

    # Set the x-ticks from 1 to 15
    plt.xticks(x_values[:-1], x_values[:-1])
    # Label the axes and add a title
    plt.xlabel('Seed spacing')
    plt.ylabel('Packets')
    # plt.yscale("log")
    plt.title(f"Additional (in)valid packets: {'2' if seed_struct_str == 'H' else '4'} byte seed")
    plt.xticks(x_values, x_values)  # Set x-tick labels as the indices
    if seed_struct_str == "I":
        plt.ylim((0, 376977775))
    # Add a legend
    plt.legend(loc="right")
    plt.autoscale()
    plt.savefig(
        f"max_hp_2_additional_valid_invalid_packets_{'H' if seed_struct_str == 'H' else 'I'}{'_payloadxor' if i else ''}.svg",
        format="svg",
        dpi=1200)
    plt.savefig(
        f"max_hp_2_additional_valid_invalid_packets_{'H' if seed_struct_str == 'H' else 'I'}{'_payloadxor' if i else ''}.pdf",
        bbox_inches="tight")
    # Show the plot
    plt.show()
    # "" "


def compare_variants_with_packets():
    res_packets = []
    for filename in ["Dorn", "aes_Dorn"]:
        for use_payload_xor in [False, True]:
            res_packets.append(
                (filename, use_payload_xor, encode(file=filename, chunk_size=40, dist=dists[0], rules=rules,
                                                   return_packets=False, repeats=repeats, id_spacing=0,
                                                   mask_id=False, use_payload_xor=use_payload_xor,
                                                   seed_struct_str="H", return_packet_error_vals=True,
                                                   store_packets=True)))
    print(res_packets)


def parse_all_files():
    # change working directory:
    # save current working directory:
    cwd = os.getcwd()
    os.chdir("./eval/err_nums")
    # open err_nums.zip and get all files:
    with zipfile.ZipFile("err_nums.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    # get all files starting with "err_nums" in the current directory:
    files = [f for f in os.listdir(".") if os.path.isfile(f) and f.startswith("err_nums") and not f.endswith(".zip")]
    # create empty dataframe:
    # df2 = pd.DataFrame(columns=["encoded_file", "use_payload_xor", "chunk_size", "id_spacing", "mask_id",
    #                            "seed_struct_str", "err_nums"])
    tmp = []
    for file in files:
        # load the csv file:
        df = pd.read_csv(file)
        # count all rows with a value >= 1.0:
        err_num = df[df["err_nums"] >= 1.0].count().tolist()[0]
        file = file.replace("err_nums_", "").split("_2023")[0]
        # split "file" at first position of "True":
        if file.find("_True_") == -1 or (file.find("_True_") > file.find("_False_") > -1):
            encoded_file, file = file.split("_False_", 1)
            use_payload_xor = False
        elif file.find("_False_") == -1 or file.find("_True_") < file.find("_False_"):
            encoded_file, file = file.split("_True_", 1)
            use_payload_xor = True
        else:
            raise ValueError("File name does not contain 'True' or 'False'!")

        # split the filename to its parts:
        filename_parts = file.split("_")
        chunk_size, id_spacing, mask_id, seed_struct_str = filename_parts
        # add results to a row of a dataframe "df2"
        tmp.append({"encoded_file": encoded_file, "use_payload_xor": use_payload_xor,
                    "chunk_size": int(chunk_size), "id_spacing": int(id_spacing), "mask_id": mask_id == "True",
                    "seed_struct_str": seed_struct_str, "err_nums": err_num})
    df2 = pd.DataFrame(tmp)
    print(df2)
    os.chdir(cwd)


def plot_entropy_vs_xor_payload_vs_rule_violating_packets(file_name: str):
    df = pd.read_csv(file_name)
    df["seed_struct_str"] = "H"
    tmp = df[(df.mask_id == False) & (df.chunk_size == 40) & (df.id_spacing == 0)]
    tmp = tmp.rename(columns={"use_payload_xor": "XOR payload"})
    tmp["file_entropy"] = tmp["file_entropy"].map(lambda x: float(x.replace("(", "").split(",")[0]))
    tmp["file_dna_entropy"] = tmp["file_dna_entropy"].map(lambda x: round(float(x.replace("(", "").split(",")[0]), 5))
    sns.barplot(x="file_dna_entropy", y="rule_violating_packets", hue="XOR payload", data=tmp)
    plt.xlabel("File entropy (DNA)")
    plt.ylabel("Rule violating packets")
    plt.title("Rule violating packets (2 byte seed)")  # , 40 chunks, 0 spacing

    # convert file_name to only contain the file name:
    file_name = file_name.split("/")[-1]
    # and remove the file ending:
    file_name = file_name.split(".")[0]
    plt.autoscale()
    plt.savefig(f"bar_rule_violating_vs_file_entropy{file_name.replace('.', '_')}.svg", format="svg", dpi=1200)
    plt.savefig(f"bar_rule_violating_vs_file_entropy.pdf{file_name.replace('.', '_')}.pdf", bbox_inches="tight")
    plt.show(block=False)


def plot_max_possible_unique_packets_per_deg(n, seed_len=2):
    # calculate
    lst = []
    max_deg = 40
    for i in range(1, max_deg + 1):
        lst.append(math.comb(n, i))

    # Create a figure and the primary y-axis
    _, ax1 = plt.subplots()

    # Plot the primary y-axis data
    ax1.plot(range(1, max_deg + 1), lst, label="Combinations", color="tab:blue")
    ax1.set_yscale("log")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Packets")
    ax1.set_title("Number of unique packets per degree")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    raptor_dist_func = norm_list(to_dist_list(raptor_dist))
    print(raptor_dist_func)

    possible_packets = math.pow(2, (seed_len * 8))
    res = [x * possible_packets for x in raptor_dist_func]
    ax1.plot(range(1, max_deg + 1), res[:max_deg], label="Possible packets (Raptor)", color="tab:cyan")
    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Plot the secondary y-axXis data
    ax2.plot(range(1, max_deg + 1), raptor_dist_func[:max_deg], label="Raptor distribution", color="tab:red")
    ax2.set_ylabel("Probability")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    # Combine legends from both axes
    # ax1.legend(loc="upper left")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax2.legend(lines, labels, loc="right")
    plt.autoscale()

    plt.savefig(f"max_unique_packets_per_deg_{n}_{seed_len}.svg", format="svg", dpi=1200)
    plt.savefig(f"max_unique_packets_per_deg_{n}_{seed_len}.pdf", bbox_inches="tight")
    plt.show(block=False)


def compare_dists(files=None, chunksize=40, seed_spacing=4, use_payload_xor=True):
    if files is None:
        files = ["Dorn"]
    folders = [f for f in os.listdir(".") if os.path.isdir(f) and f.startswith("results_")]

    # create a csv file:
    with open(f"eval/dist_compare_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            ["dist_variation", "file", "chunk_size", "dist", "id_spacing", "use_payload_xor", "mask_id",
             "seed_struct_str", "avg_error", "variance_error", "rule_violating_packets", "number_of_chunks",
             "file_entropy", "file_dna_entropy", "avg_overhead", "unique_packets", "avg_unrecovered",
             "overhead", "clean_avg_error", "clean_deg_len", "non_unique_packets"])
        for cmp_folder in folders:
            j = None
            with open(f"{cmp_folder}/diff_opt_state.json", "r") as f:
                j = json.load(f)
            dist_func = j["finished_prev_best"]["dist_lst"]
            # create all packets for the dist:
            dist = Distribution.Distribution(dist_func)

            for file in files:
                _, overhead, clean_avg_error, clean_deg_len, non_unique_packets, _ = dist.compute_fitness(
                    [file], 1, chunksize)
                _ = dist.calculate_error_value()
                res = encode(file, chunksize, dist.to_raptor_list(), repeats=1, return_packets=False,
                             return_packet_error_vals=False, id_spacing=seed_spacing, use_payload_xor=use_payload_xor)
                overhead_avg, degree_dict, unique_packets, unrecovered_avg = res[0]
                packet_error_vals, number_of_chunks = res[1]
                packet_error_vals = pd.array(packet_error_vals, dtype=np.double)
                print(f"{cmp_folder} - avg error: {np.average(packet_error_vals)}")
                # write the results to the csv file:
                csv_writer.writerow(
                    [cmp_folder, file, chunksize, dist.dist_lst, seed_spacing, use_payload_xor, False, "H",
                     np.average(packet_error_vals), np.var(packet_error_vals),
                     len(packet_error_vals[packet_error_vals > 1.0]), number_of_chunks,
                     calculate_entropy(file, convert_to_dna=False), calculate_entropy(file), overhead_avg,
                     unique_packets, unrecovered_avg, overhead, clean_avg_error, clean_deg_len,
                     non_unique_packets])


def create_new_param_compare_csv():
    output_filename = f"param_compare_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(output_filename, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            ["file", "chunk_size", "dist", "id_spacing", "use_payload_xor", "mask_id", "seed_struct_str", "avg_error",
             "variance_error", "rule_violating_packets", "number_of_chunks", "file_entropy", "file_dna_entropy"])

        # Create a list of all combinations of parameters for multiprocessing
        combinations = [(file, chunk_size, dist, id_spacing, use_payload_xor, xor_seed, seed_struct_str)
                        for file in files
                        for chunk_size in chunk_sizes
                        for dist in dists
                        for id_spacing in id_spacings
                        for use_payload_xor in [True, False]
                        for xor_seed in [True, False]
                        for seed_struct_str in seed_struct_strs]

        num_workers = multiprocessing.cpu_count() - 1

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process_combination, combinations)

        for result in results:
            csv_writer.writerow(result)


if __name__ == "__main__":
    """
    create_new_param_compare_csv()
    plot_entropy_vs_xor_payload_vs_rule_violating_packets("eval/param_compare/param_compare_2023-09-08_16-01-23.csv")

    compare_dists(files=["logo_mosla_bw.bmp", "Dorn", "aes_Dorn"], chunksize=40,
                  seed_spacing=4, use_payload_xor=True)

    plot_max_possible_unique_packets_per_deg(50)
    plot_max_possible_unique_packets_per_deg(100)
    plot_max_possible_unique_packets_per_deg(500)
    plot_max_possible_unique_packets_per_deg(5000)
    plot_max_possible_unique_packets_per_deg(50000)

    plot_max_possible_unique_packets_per_deg(50, 4)
    plot_max_possible_unique_packets_per_deg(100, 4)
    plot_max_possible_unique_packets_per_deg(500, 4)
    plot_max_possible_unique_packets_per_deg(5000, 4)
    plot_max_possible_unique_packets_per_deg(50000, 4)

    parse_all_files()
    """
    get_packets("I")
    # get_packets("I")
    """
    # depending on the file content, some plots may fail, this is expected and should not be a problem
    try:
        create_graphs("eval/param_compare/param_compare_2023-09-07_15-58-41.csv")  # "param_compare_2023-08-17_10-31-15.csv")
    except Exception as e:
        print(e)
    try:
        create_graphs("eval/param_compare/param_compare_2023-09-08_10-10-36.csv")  # "param_compare_2023-08-17_10-31-15.csv")
    except Exception as e:
        print(e)

    plot_entropy_vs_xor_payload_vs_rule_violating_packets("eval/param_compare/param_compare_2023-09-08_16-01-23.csv")
    plot_err_nums([("plain IDs", "eval/err_nums/err_nums_Dorn_False_40_0_False_H_2023-09-07_10-35-44.csv"),
                   ("XOR shuffled IDs", "eval/err_nums/err_nums_Dorn_False_40_0_True_H_2023-09-07_10-35-05.csv")])

    " ""
    # Code to generate a new "param_compare" csv file (configure at the top of the file!):
    """
