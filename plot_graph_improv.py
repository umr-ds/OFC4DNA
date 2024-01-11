import json

import matplotlib.pyplot as plt
import pandas as pd

from Helper import calculate_entropy
import seaborn as sns

sns.set_theme()


def create_graphs(json_filename, inital_title=""):
    data = None
    with open(json_filename) as f:
        data = json.load(f)
    # create a dataframe from the json data:
    df = pd.DataFrame(data["pop"])
    df = df.drop(columns=["dist_lst", "degree_errs"])
    # remove all columns ending in "_fac":
    df = df.loc[:, ~df.columns.str.endswith("_fac")]
    for x in df.columns:
        try:
            df[x].plot.line()
            plt.title(inital_title + " - " + x)
            plt.show(block=False)
            plt.clf()
        except TypeError:
            pass

    df.plot.line()
    plt.title(inital_title)
    plt.show(block=False)

    # remove column "clean_deg_len":
    df = df.drop(columns=["clean_deg_len", "non_unique_packets"])
    # plot all columns as a line in a line-plot:
    df.plot.line()
    plt.title(inital_title + " - clean_deg_len and non_unique_packets")
    plt.show(block=False)
    try:
        df = df.drop(columns=["avg_unrecovered"])
    except KeyError:
        pass
    df.plot.line()
    plt.title(inital_title + " - clean_deg_len and non_unique_packets and avg_unrecovered")
    plt.show(block=False)


if __name__ == "__main__":
    bmp_low_entropy = ["logo_mosla_bw.bmp"]
    image_high_entropy = ["logo.jpg", "logo_mosla_rgb.png", "Marburg_Wall_CC0.png"]
    compress_encrypt_high_entropy = ["Dorn.zip", "aes_Dorn", "aes_ecb_Dorn"]
    text_medium_entropy = ['Dorn', 'LICENSE', 'Rapunzel', 'Rothkäppchen', 'Sneewittchen']
    text_medium_high_entropy = ["Dorn.pdf", "lorem_ipsum100k.doc"]

    input_files = bmp_low_entropy + image_high_entropy + compress_encrypt_high_entropy + text_medium_entropy + \
                  text_medium_high_entropy
    res = []
    for input_file in input_files:
        tmp = calculate_entropy(input_file, convert_to_dna=False)
        val, tmpp = tmp
        print(f"{input_file} : {tmp}")
        tmp2 = calculate_entropy(input_file, convert_to_dna=True)
        val2, tmpp2 = tmp2
        print(f"DNA: {input_file} : {tmp2}")
        tmpp["File"] = input_file
        tmpp["Entropy"] = val
        res.append(tmpp)
        tmpp2["File"] = f"{input_file}_DNA"
        tmpp2["Entropy"] = val2
        res.append(tmpp2)
    df = pd.DataFrame.from_records(res)
    filtered_df = df[df["File"].str.contains("DNA")]
    filtered_df = filtered_df.copy()
    filtered_df["File"] = filtered_df["File"].str.replace("_DNA", "")
    filtered_df.plot.bar(x="File", y="Entropy")
    plt.xticks(rotation=85)
    plt.subplots_adjust(bottom=0.42)
    plt.legend([])
    plt.ylabel("Entropy")
    plt.title("Entropy of DNA-sequences")
    plt.savefig("file_entropy_DNA.svg", format="svg", dpi=1200)
    plt.savefig("file_entropy_DNA.pdf", bbox_inches="tight")
    plt.show()
    print(df)
    """
    # get list of all subfolders in "final_results":
    subfolders = [f.path for f in os.scandir("final_results") if f.is_dir()]
    for subfolder in subfolders:
        for sub_subfolder in [f.path for f in os.scandir(subfolder) if f.is_dir()]:
            if "diff_opt_state.json" in os.listdir(sub_subfolder):
                create_graphs(f"{sub_subfolder}/diff_opt_state.json", inital_title=subfolder)
            elif "evo_opt_state.json" in os.listdir(sub_subfolder):
                create_graphs(f"{sub_subfolder}/evo_opt_state.json", inital_title=subfolder)
    """
