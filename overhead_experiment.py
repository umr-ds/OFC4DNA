import matplotlib
import seaborn as sns
import base64
import glob
import json
import os
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from NOREC4DNA.norec4dna import get_error_correction_decode
from encode_decode_experiment import get_rs_sym, get_dist, get_payload_xor, get_seed_spacing, get_num_chunks, \
    decode_from_fasta

sns.set(font_scale=1.25)
sns.set_style("ticks", {'axes.grid': True})
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
matplotlib.rcParams.update({'figure.autolayout': True}),

def load_fasta(fasta_file):
    """
    Loads fasta file and returns a dictionary of sequences
    """
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_name = line.strip()
                fasta_dict[seq_name] = ''
            else:
                fasta_dict[seq_name] += line.strip()
    return fasta_dict


def try_decode(folder, dna_fountain_dir):
    def decode_ez(file, dna_fountain_dir):
        # f"{current_dir}/datasets/out/ez_{filename}_nc{chunks}.fasta"
        current_dir = os.getcwd()
        # get filename only from path given in file using python libs:
        filename = os.path.basename(file)
        # get the path:
        path = os.path.dirname(file)
        # get the full path:
        abs_file = os.path.abspath(file)
        chunks = int(file.split("_nc")[1].split(".")[0])
        # run the command python decode.py:
        command = f"cd {dna_fountain_dir} && " \
                  f"source venv/bin/activate && " \
                  f"python decode.py -f {abs_file} -n {chunks} -m 3 --gc 0.10 --rs 2 --delta 0.05 --c_dist 0.1 --out {filename}.result_ez --fasta && " \
                  f"cp {filename}.fasta {current_dir}/datasets/out/ez_{filename}.fasta && " \
                  f"cd {current_dir}"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                                   executable="/bin/bash")
        result = process.communicate()
        res = int(result[0].decode("utf-8").split(" chunks are done")[0].split(", ")[1])
        if res == chunks:
            # TODO: we might want to compare the result with the original file!
            return True
        else:
            return False

    def decode_grass(file):
        current_dir = os.getcwd()
        # get filename only from path given in file using python libs:
        file_name = os.path.basename(file)
        # get the path:
        path = os.path.dirname(file)
        # get the full path:
        full_path = os.path.abspath(file)
        # get the number of blocks:
        blocks = file.split("_blocks")[1].split(".")[0]
        # open the output file and get the length of the first line (without newline):
        fasta = load_fasta(full_path)
        # convert to a file with only the sequences:
        dna_file = f"/tmp/grass_{file_name.replace('.fasta', '.dna')}"
        with open(dna_file, "w") as o_:
            for key, value in fasta.items():
                o_.write(f"{value.strip()}\n")
        # decode using grass code using the external executable "./texttodna --decode --input <file>  --output <file>.fasta":
        process = subprocess.Popen(
            f"cd datasets/grass && ./texttodna --decode --input {dna_file} --output {file_name}.result_grass && cd {current_dir}",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
        result = process.communicate()
        # TODO: compare with original file or parse result!
        return result

    # try to decode each file and save the result in a csv:
    files = glob.glob(f"{folder}/*.fasta")
    csv_line = "file,dist_name,use_payload_xor,seed_spacing,static_number_of_chunks,error_rate,success,decoded_data\n"
    for file in files:
        if file.startswith("grass"):
            success = decode_grass(file)
            csv_line += f"{file},grass,grass,grass,grass,{error_rate},{success},grass\n"
        elif file.startswith("ez"):
            success = decode_ez(file, dna_fountain_dir)
            csv_line += f"{file},ez,ez,ez,ez,{error_rate},{success},ez\n"
        else:
            rs_sym = get_rs_sym(file)
            error_correction = get_error_correction_decode("reedsolomon", rs_sym)
            dist, dist_name = get_dist(file)
            use_payload_xor = get_payload_xor(file)
            seed_spacing = get_seed_spacing(file)
            static_number_of_chunks = get_num_chunks(file)
            error_rate = float(file.split("error_")[1].split(".fasta")[0])
            print(
                f"Current file and settings: {file}, {dist_name}, {use_payload_xor}, {seed_spacing}, {static_number_of_chunks}")
            # try:
            try:
                res = decode_from_fasta(file, number_of_chunks=static_number_of_chunks, dist=dist,
                                        error_correction=error_correction,
                                        use_seed_xor=True, use_payload_xor=use_payload_xor, seed_spacing=seed_spacing,
                                        use_headerchunk=False)
            except Exception as e:
                res = (False, "")
                # raise e
            try:
                r = res[1].encode()
            except:
                r = res[1]
            csv_line += f"{file},{dist_name},{use_payload_xor},{seed_spacing},{static_number_of_chunks},{error_rate},{res[0]},{base64.b64encode(r)}\n"

    with open(f"{folder}/error_results.csv", "w") as o_:
        o_.write(csv_line)

        # except Exception as e:
        #    continue


def fasta_random_permutation(in_file, out_file):
    fasta = load_fasta(in_file)
    # randomly permute the sequences:
    import random
    keys = list(fasta.keys())
    random.shuffle(keys)
    with open(out_file, "w") as o_:
        for key in keys:
            o_.write(f"{key.strip()}\n{fasta[key].strip()}\n")


def gen_mutations(base_folder):
    files = glob.glob(f"{base_folder}/*.fasta", recursive=False)
    for file in files:
        print(f"Now processing file: {file}")
        file_name = os.path.basename(file)
        # get the path:
        path = os.path.dirname(file)
        # get the full path:
        full_path = os.path.abspath(file)
        if file.startswith("grass"):
            continue  # the overhead should be the same for each permutation...
        for repeat in range(4):
            new_ending = f"_permuted_{repeat}.fasta"
            fasta_random_permutation(file, f"{base_folder}/permutations/{file_name.replace('.fasta', new_ending)}")
        # copy the original file - as the orignial version is also a valid permutation:
        shutil.copy(file, f"{base_folder}/permutations/{file_name.replace('.fasta', '_permuted_original.fasta')}")


def calc_gc_content(fasta):
    gc_content = []
    for key, value in fasta.items():
        gc_content.append((value.count("G") + value.count("C")) / len(value))
    return gc_content


def calc_homopolymer(fasta, hp_len=2):
    hp_content = []
    for key, value in fasta.items():
        row_hp = 0
        row_hp += value.count("A" * hp_len)
        row_hp += value.count("T" * hp_len)
        row_hp += value.count("G" * hp_len)
        row_hp += value.count("C" * hp_len)
        hp_content.append(row_hp)
    return hp_content


def analyze_files(base_folder):
    files = glob.glob(f"{base_folder}/*.fasta", recursive=False)
    res = {}
    for file in files:
        print(f"Now processing file: {file}")
        file_name = os.path.basename(file)
        # get the path:
        path = os.path.dirname(file)
        # get the full path:
        full_path = os.path.abspath(file)
        fasta = load_fasta(file)
        gc_content = calc_gc_content(fasta)
        hps_two = calc_homopolymer(fasta)
        hps_three = calc_homopolymer(fasta, hp_len=3)
        # write the results to a csv:
        res[file_name] = {"gc_contents": gc_content, "hp_contents_two": hps_two, "hp_contents_three": hps_three}
    # store res as json:
    with open(f"{base_folder}/analysis.json", "w") as o_:
        json.dump(res, o_)
    return res


def plot_hp(df, hp_len):
    # filter for files that have ".txt" in it:
    # df = df[df["filename"].str.contains(".txt")]
    if hp_len == 2:
        hp_meth = "hp_two"
    elif hp_len == 3:
        hp_meth = "hp_three"
    else:
        raise ValueError("hp_len must be 2 or 3")
    # get all unique methods:
    # methods = df["method"].unique()
    # files = df["filename"].unique()
    # df['Filetype'] = df['Filetype'].apply(lambda x: x.split(".")[1])
    # to_plot = {}
    # for method in methods:
    #    print(f"Method: {method}")
    #    print(df[df["method"] == method][hp_meth].describe())
    #    to_plot[method.replace("_", " ")] = df[df["method"] == method][hp_meth]
    ## group by method and filename:
    # grouped = df.groupby(["method", "filename"])

    # create a violin plot:
    # fig, ax = plt.subplots(figsize=(8, 6))
    # plot to_plot using the key as the x label:
    # sns.violinplot(data=df, x="Method", y=hp_meth, hue="Filetype", ax=ax, bw_method='scott')
    sns.set_style("darkgrid")
    fig = sns.catplot(kind='boxen', x="Filetype", y=hp_meth, hue="Method", data=df, height=6, aspect=2)
    #plt.xticks(rotation=45, ha='right')
    # sns.catplot(kind='violin', data=df, x="method", y=hp_meth, hue="filename_group", ax=ax, bw_method='scott')
    plt.ylabel(f'Number of Homopolymers of length {hp_len}')
    plt.xlabel('Method')
    plt.title(f'Number of homopolymers of length {hp_len} over all files')
    plt.grid(True)
    # ax.set_xticks(range(len(to_plot)))
    # ax.set_xticklabels([x for x in to_plot.keys()])
    # print([x for x in to_plot.keys()])
    plt.savefig(f"hp_{hp_len}_plot_same_len.pdf", bbox_inches="tight")
    plt.savefig(f"hp_{hp_len}_plot_same_len.svg")
    plt.show(block=False)


def plot_gc(df):
    sns.set_style("darkgrid")
    fig = sns.catplot(kind='boxen', x="Filetype", y="GC content", hue="Method", data=df, height=6, aspect=2,
                      )
    #plt.xticks(rotation=45, ha='right')

    # sns.violinplot(data=to_plot, ax=ax, bw_method='scott')
    plt.ylabel('GC content')
    plt.xlabel('Filetype')
    plt.title('GC content distribution over all files')
    plt.grid(True)
    # ax.set_xticks(range(len(to_plot)))
    # ax.set_xticklabels([x for x in to_plot.keys()])
    plt.savefig("gc_plot_same_len.pdf", bbox_inches="tight")
    plt.savefig("gc_plot_same_len.svg")
    plt.show(block=False)


if __name__ == "__main__":
    current_dir = os.getcwd()
    base_folder = f"{current_dir}/datasets/out"
    """
    #res = analyze_files(base_folder)
    a = json.load(open(f"{base_folder}/analysis.json", "r"))
    # todo: plot the various aspects of the analysis using seaborn:
    gc_cont = {key: value["gc_contents"] for key, value in a.items()}
    res = []
    for key, value in a.items():
        if key.startswith("grass"):
            filename = key.split("_")[1]
            method = "Grass"
            nc = key.split("_blocks")[1].split(".")[0]
            rs = "-1"
        elif key.startswith("ez"):
            filename = key.split("_")[1]
            method = "DNA Fountain"
            nc = key.split("_nc")[1].split(".")[0]
            rs = "2"
        else:
            filename = key.split("_")[0]
            tmp = key.replace(filename, "")
            method, rs = tmp[1:].split("_rs")
            rs = rs.split("_")[0]
            if "_baseline" in key:
                method = "Raptor (baseline)"
                nc = tmp.split("_nc")[1].split("_")[0]
            else:
                nc = tmp.split("_nc")[1].split(".")[0]
            method = method.replace("_", " ")[1:]

        for i in range(len(value["gc_contents"])):
            row = {"filename": filename, "method": method, "rs": rs, "nc": nc, "gc_content": value["gc_contents"][i],
                   "hp_two": value["hp_contents_two"][i], "hp_three": value["hp_contents_three"][i]}
            res.append(row)

    # build a dataframe with this data:

    df = pd.DataFrame(res)
    df.to_csv("gc_hp_hp_results.csv")
    """
    df = pd.read_csv("gc_hp_hp_results.csv")
    # replace every method as following: "mp low entropy evo dist"-> "low entropy evo" "vo compress encrypt high entropy dist" -> "high entropy evo", "aptor dist" -> "Raptor", "aptor (baseline)" -> "Raptor (baseline)":
    df["method"] = df["method"].apply(lambda x: x.replace("mp low entropy evo dist", "Low entropy - evo").replace(
        "vo compress encrypt high entropy dist", "High entropy - evo").replace("aptor dist", "Raptor").replace(
        "aptor (baseline)", "Raptor (baseline)").replace("dist", ""))
    # df.to_csv("gc_hp_hp_results_correct.csv")
    df = df.rename(columns={"method": "Method", "gc_content": "GC content", "filename": "Filetype"})
    # filter out any row with "Raptor" in its method:
    df = df[~df["Method"].isin(["Raptor"])]
    # for each file and method, limit the number of rows to the same as there are rows for the "DNA Fountain" method:
    # iterate over all files:
    """
    files = df["Filetype"].unique()
    for file in files:
        # get the number of rows for the "DNA Fountain" method:
        num_rows = df[df["Method"].isin(["DNA Fountain"])].shape[0]
        # get the rows that are not "DNA Fountain" and limit them to the same number of rows:
        to_limit = df[df["Filetype"].isin([file])]
        to_limit = to_limit.head(num_rows)
        # update the dataframe:
        df = df[~df["Filetype"].isin([file])]
        df = pd.concat([df, to_limit])
    #"""

    df['Filetype'] = df['Filetype'].apply(lambda x: x.split(".")[1])

    # only use the first 1000 rows:
    # df = df.head(10000)
    # df['Filetype'] = df['Filetype'].str[5:]
    """
    methods = df["Method"].unique()
    
    for method in methods:
        print(f"Method: {method}")
        filter_df = df[df["Method"] == method]
        print("------------------- GC-C -------------------")
        print(filter_df["GC content"].describe())
        print("------------------- HP 2 -------------------")
        print(filter_df["hp_two"].describe())
        print("------------------- HP 3 -------------------")
        print(filter_df["hp_three"].describe())
    """
    plot_gc(df)
    plot_hp(df, hp_len=2)
    plot_hp(df, hp_len=3)


    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.violinplot(data=df, x="filename", y="gc_content", ax=ax)
    # gen_mutations(base_folder, num_repeats=4)
    # """