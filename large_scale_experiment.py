import subprocess

from Helper import scale_to, encode
from NOREC4DNA.invivo_window_decoder import load_fasta
import os
import numpy as np
import pandas as pd
import base64
import glob
from NOREC4DNA.norec4dna import get_error_correction_decode, RU10Decoder, RU10Encoder, get_error_correction_encode
from NOREC4DNA.norec4dna.distributions.RaptorDistribution import RaptorDistribution
from NOREC4DNA.optimizer.optimization_helper import diff_list_to_list
from cluster_exp import raptor_dist, bmp_low_entropy_evo_dist, bmp_low_entropy_diff_dist, \
    evo_compress_encrypt_high_entropy_dist, diff_compress_encrypt_high_entropy_dist
from NOREC4DNA.norec4dna.helper.RU10Helper import intermediate_symbols
from NOREC4DNA.norec4dna.rules.FastDNARules import FastDNARules
from NOREC4DNA.find_minimum_packets import main as mp_gen_min_packets
import random
from copy import deepcopy
import json
import requests


def encode_to_fasta(filename, number_of_chunks, error_correction, use_seed_xor, use_payload_xor, seed_spacing,
                    use_headerchunk, in_dist, seed_len_str="I", out_file_prefix="out_file", chunk_size=0):
    dist = RaptorDistribution(number_of_chunks)
    dist.f = in_dist
    dist.d = [x for x in range(0, 41)]
    encoder = RU10Encoder(filename, number_of_chunks, dist, insert_header=use_headerchunk, pseudo_decoder=None,
                          rules=FastDNARules(), error_correction=error_correction, packet_len_format="I",
                          crc_len_format="I", chunk_size=chunk_size,
                          number_of_chunks_len_format="I", id_len_format=seed_len_str,
                          save_number_of_chunks_in_packet=False,
                          mode_1_bmp=False, prepend="", append="", drop_upper_bound=1.0, keep_all_packets=False,
                          checksum_len_str=None, xor_by_seed=use_payload_xor, mask_id=use_seed_xor,
                          id_spacing=seed_spacing)
    encoder.set_overhead_limit(3.0)
    encoder.encode_to_packets()
    encoder.encodedPackets = sorted(encoder.encodedPackets, key=lambda x: x.error_prob)
    return encoder


def decode_from_fasta(filename, number_of_chunks, dist, error_correction, use_seed_xor, use_payload_xor, seed_spacing,
                      use_headerchunk, seed_len_str="I"):
    decoder = RU10Decoder(file=filename, error_correction=error_correction, use_headerchunk=use_headerchunk,
                          static_number_of_chunks=number_of_chunks, checksum_len_str=None,
                          xor_by_seed=use_payload_xor, mask_id=use_seed_xor, id_spacing=seed_spacing)

    decoder.distribution = RaptorDistribution(number_of_chunks)
    # update distribution.f:
    decoder.distribution.f = dist
    decoder.distribution.d = [x for x in range(0, 41)]
    decoder.number_of_chunks = number_of_chunks
    _, decoder.s, decoder.h = intermediate_symbols(number_of_chunks, decoder.distribution)
    decoder.createAuxBlocks()
    decoder.progress_bar = decoder.create_progress_bar(number_of_chunks + 0.02 * number_of_chunks)
    res = decoder.decodeFile(id_len_format=seed_len_str)
    print(f"Success: {res}")
    res_data = decoder.saveDecodedFile(last_chunk_len_format="", null_is_terminator=False, print_to_output=False,
                                       partial_decoding=True)
    return res, res_data, decoder


# encoder = encode_to_fasta("sleeping_beauty", 289, get_error_correction_encode("reedsolomon", 2), False, True, 2, False,
#                          raptor_dist)
# encoder.save_packets_fasta(
#    f"out_file.fasta",
#    seed_is_filename=True)
# res, res_data, decoder = decode_from_fasta("out_file.fasta", 289, raptor_dist,
#                                           get_error_correction_decode("reedsolomon", 2), False, True, 2,
#                                           False)


def get_dist(filename):
    if "raptor" in filename:
        return raptor_dist, "raptor"
    elif "low_entropy_evo" in filename:
        return bmp_low_entropy_evo_dist, "low_entropy_evo"
    elif "low_entropy_diff" in filename:
        return bmp_low_entropy_diff_dist, "low_entropy_diff"
    elif "evo_compress_encrypt" in filename:
        return evo_compress_encrypt_high_entropy_dist, "evo_compress_encrypt"
    elif "diff_compress_encrypt" in filename:
        return diff_compress_encrypt_high_entropy_dist, "diff_compress_encrypt"
    else:
        raise ValueError("Unknown dist")


def get_rs_sym(file):
    if "rs" not in file:
        raise ValueError("Unknown rs")
    return int(file.split("rs")[1].split("_")[0])


def get_seed_spacing(filename):
    if "seedspacing" not in filename:
        return 0
    else:
        return int(filename.replace("_payloadxor", "").split("seedspacing")[1].split(".")[0])


def get_payload_xor(filename):
    return "_payloadxor" in filename


def get_num_chunks(filename):
    return filename.split("nc")[1].split("_")[0]


# load mesa_config.json into a dict:
config = json.load(open("mesa_config.json"))


def get_mesa_errors_seqs(sequence, error_multiplier=1.0, apikey="IgGD6Cfdlnqa4tUungucZpKp3hfYkt1IDqg0Bn3BxEE"):
    def apply_multiplier(config, multiplier):
        for err_rule in ["homopolymer_error_prob", "gc_error_prob"]:
            for key in range(len(config[err_rule]["data"])):
                config[err_rule]["data"][key]["y"] *= min(100.0, multiplier)
        return config

    config["key"] = apikey
    config["asHTML"] = False
    config["sequence"] = sequence
    mesa_config = apply_multiplier(config, error_multiplier)
    res = requests.post("http://127.0.0.1:5000/api/all", json=mesa_config)
    return res.json()[sequence]["res"]["modified_sequence"].replace(" ", "")


def random_errors(names_2_sequences, error_rate=0.013, seq_drop_rate=0.00001, max_nts=1000000000):
    # apply random errors to the sequences such that the overall per base error rate is error_rate. An error may not lead to the correct base!:
    total_nts = 0
    error_rate = min(1.0, error_rate)
    error_rate = max(0.0, error_rate)
    out_name2seqs = {}
    subs = 0
    ins = 0
    dels = 0
    drop = 0
    for name, seq in names_2_sequences.items():
        total_nts += len(seq)
        if random.random() < seq_drop_rate:
            drop += 1
            continue  # dont add the sequence to the output list
        for i in range(len(seq)):
            if random.random() < error_rate:
                mode = np.random.choice(["sub", "ins", "del"], 1, p=[0.8, 0.1, 0.1])[0]
                if mode == "sub":
                    seq = seq[:i] + random.choice(list({"A", "T", "G", "C"}.difference(seq[i]))) + seq[i + 1:]
                    subs += 1
                elif mode == "ins":
                    seq = seq[:i] + random.choice(list({"A", "T", "G", "C"})) + seq[i:]
                    ins += 1
                else:
                    seq = seq[:i] + " " + seq[i + 1:]
                    dels += 1
        out_name2seqs[name] = seq.replace(" ", "")
        if total_nts >= max_nts:
            # ensure we only use as much sequence as the grass code produced!
            break
    return out_name2seqs, (subs, ins, dels, drop)


# get_mesa_errors_seqs("ACACCAGTTGC")
# random_errors({">seq1": "ACGGCTCGCATACG", ">seq2": "AAAAAAAA"}, 0.01, -0.0001)


def modify_seq(original, pos, probs, results, counter, seqcount, base=None):
    pos_sub = []
    pos_ins = []
    pos_del = []
    sub_val = probs["substitution"]
    del_val = sub_val + probs["deletion"]
    ins_val = del_val + probs["insertion"]
    for p in pos:
        ran_num = np.random.randint(0, 100)
        if ran_num <= sub_val:
            pos_sub.append(p)
        elif ran_num <= del_val:
            pos_del.append(p)
        else:
            pos_ins.append(p)
    # print(pos_sub, pos_del, pos_ins)
    results[str(counter)][str(seqcount)]["sub_pos"] = pos_sub
    results[str(counter)][str(seqcount)]["ins_pos"] = pos_ins
    results[str(counter)][str(seqcount)]["del_pos"] = pos_del
    modified = deepcopy(original)
    if pos_sub:
        modified = substitutions(modified, pos_sub)
    if pos_ins:
        modified = insertions(modified, pos_ins)
    if pos_del:
        modified = deletions(modified, pos_del)
    return (modified, len(pos_sub), len(pos_ins), len(pos_del))


def modify_file(in_path, out_path, probs, results, counter, err_list, weights):
    linecount = 0
    seqcount = 0
    errcount = 0
    subcount = 0
    inscount = 0
    delcount = 0
    modseqscount = 0
    with open(in_path, "r") as inp, open(out_path, "w") as out:
        while True:
            line = inp.readline()
            linecount += 1
            if not line:
                break
            if linecount % 2 != 0:
                out.write(line)
            else:
                ori = line.strip()
                num_errs = np.random.choice(err_list, 1, p=weights)[0]  # p=[0.4, 0.3, 0.2, 0.05, 0.05]
                errcount += num_errs
                results[str(counter)][str(seqcount)] = dict()
                results[str(counter)][str(seqcount)]["num_errs"] = int(num_errs)
                if not num_errs:
                    out.write(line)
                else:
                    modseqscount += 1
                    pos = random.sample(range(0, len(ori)), num_errs)
                    results[str(counter)][str(seqcount)]["error_pos"] = pos
                    seq, nsub, nins, ndels = modify_seq(ori, pos, probs, results, counter, seqcount)
                    subcount += nsub
                    inscount += nins
                    delcount += ndels
                    out.write(seq + "\n")
            seqcount += 1
        results[str(counter)]["number_of_errors"] = int(errcount)
        results[str(counter)]["number_of_modified_seqs"] = int(modseqscount)
        results[str(counter)]["number_of_substitutions"] = int(subcount)
        results[str(counter)]["number_of_insertions"] = int(inscount)
        results[str(counter)]["number_of_deletions"] = int(delcount)


def substitutions(original, pos, base=None):
    modified = deepcopy(original)
    for ele in pos:
        if not base:
            base = random.choice(list({"A", "T", "G", "C"}.difference(original[ele])))
        modified = modified[:ele] + base + modified[ele + 1:]
    return modified


def insertions(original, pos, base=None):
    modified = deepcopy(original)
    shift = 0
    pos.sort()
    for ele in pos:
        if not base:
            base = random.choice(list({"A", "T", "G", "C"}))
        modified = modified[:ele + shift] + base + modified[ele + shift:]
        shift += 1
    return modified


def deletions(original, pos):
    modified = deepcopy(original)
    shift = 0
    pos.sort()
    for ele in pos:
        modified = modified[:ele - shift] + modified[ele - shift + 1:]
    return modified


"""
in_folder = "clusts/"
in_file = "cs_23_I_max_2_hp_10_gc_opt_sleeping_beauty_150_raptor_seedspacing0_payloadxor.fasta"
in_path = f"{in_folder}{in_file}"
out_path = f"{in_folder}error_test/{in_file}"
length = 116
probs = {"substitution": 80, "insertion": 10, "deletion": 10}
err_list = [0, 1, 2, 3, 4, 5, 6, 7]
weights = [0.15, 0.15, 0.25, 0.2, 0.1, 0.06, 0.05, 0.04]  # weights = [0.35, 0.25, 0.15, 0.15, 0.05, 0.03, 0.01, 0.01]
iterations = 1
results = dict()
results["params"] = dict()
results["params"]["err_type"] = probs
results["params"]["num_errs"] = err_list
results["params"]["errs_weights"] = weights
counter = 0
results[str(counter)] = dict()
print("start")
start = time()
modify_file(in_path, out_path, probs, results, counter, err_list, weights)
end = time()
print(end - start)
with open("test_split.yaml", "w") as o_:
    yaml.dump(results, o_)
"""

bmp_low_entropy_evo_dist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004971830584445606, 0.021156042715781434, 0.0,
                            0.0, 0.046598859494963854, 0.04050492331421386, 0.0, 0.04809096147307328, 0.0, 0.0,
                            0.06925547416308825, 0.04532100335793773, 0.0, 0.0, .042668615395303304, 0.0,
                            0.05816837349875859, 0.03192880067497832, 0.03802025041832258, .044968466096850894,
                            0.03558473680486723, 0.029744521403035355, 0.05121639254344267, 0.0374629392718176,
                            0.03647931445297305, 0.047747672087969276, 0.05476667945725357, 0.04528494162225304,
                            0.07783895885703183, 0.04164461679912541, 0.03467221400441464, 0.01590341150809876]
bmp_low_entropy_evo_dist = scale_to(diff_list_to_list(bmp_low_entropy_evo_dist), 1048576)

bmp_low_entropy_diff_dist = [0.0010843093439073137, 0.007585735667248767, 0.0028807487219476126, 0.0008633477404769521,
                             0.04285990733936592, 0.013198093021637457, 0.0006570240115704037, 0.0506976852299126,
                             0.03987067627381218, 0.03855953667866308, 0.04357898806706369, 0.007757985783490819,
                             0.046984313365351686, 0.010214050507630095, 0.04493830021405755, 0.023652275921144446,
                             0.00464181052056198, 0.04553260862747855, 0.02201611233315065, 0.007921113084693305,
                             0.004631002645290646, 0.03662221952686662, 0.0027091735702709442, 0.04194512702766884,
                             0.022288518262633495, 0.01999377137953349, 0.0160611981296819, 0.0500439906111607,
                             0.0390939301508127, 0.046337594000175834, 0.021496185996437402, 0.009400421039175594,
                             0.042341290846698146, 0.0031643277474615373, 0.02233346996464411, 0.02763039006158188,
                             0.017722380330575195, 0.052653732160742844, 0.04302024630806095, 0.02501640778736209]
bmp_low_entropy_diff_dist = scale_to(diff_list_to_list(bmp_low_entropy_diff_dist), 1048576)

evo_compress_encrypt_high_entropy_dist = [0.0013594048887460395, 0.024392575968035388, 0.015914749333043713,
                                          0.018695909592454388, 0.023358685184162963,
                                          0.03236783551192264, 0.018216166247821023, 0.03815209530861828,
                                          0.02626279541362586, 0.026771738033088587,
                                          0.02240981497836683, 0.02900653795560873, 0.029469219106741655,
                                          0.03148959262239247, 0.026607500636491237,
                                          0.029993377912122702, 0.02729437495737362, 0.026240710500059775,
                                          0.029520514187713967, 0.03144957623804716,
                                          0.026660139657590207, 0.0, 0.036582102310668094, 0.023023044857186127,
                                          0.029136110541167947, 0.0405114142787493,
                                          0.02593307536238757, 0.025394969853798632, 0.019287711635823702,
                                          0.024887158571412053, 0.024965334073119294,
                                          0.029143289300996692, 0.03509571086315818, 0.018402552246391504,
                                          0.0270945738944279, 0.023436914977565768,
                                          0.015901655364440288, 0.032297209895409, 0.028740382914874504,
                                          0.004533474824396502]
evo_compress_encrypt_high_entropy_dist = scale_to(diff_list_to_list(evo_compress_encrypt_high_entropy_dist), 1048576)

diff_compress_encrypt_high_entropy_dist = [0.0, 0.027742190569628983, 0.041381430387865605, 0.024661226394784533,
                                           0.03168760035607533, 0.042586268544093045, 0.050172438239758727,
                                           0.02240595277423825, 0.022744587962995572, 0.02219616498815092, 0.0,
                                           0.0250794427132504, 0.02340166233385067, 0.0354141468356474,
                                           0.005684106293662012, 0.04266178285648165, 0.004800692990094652,
                                           0.02111713250802425, 0.04036783935327176, 0.045008748835130855,
                                           0.010439857196132265, 0.020439399221880886, 0.04246549723441552, 0.0,
                                           0.019110068800104227, 0.02766843648291822, 0.027110395639511172,
                                           0.03395117226609973, 0.016734711153209517, 0.016655637276022293,
                                           0.005009021579709849, 0.04196046472591439, 0.009162780801450403,
                                           0.01917442849327642, 0.02673033190145107, 0.014242757680173164,
                                           0.02024259118571735, 0.02817669668334466, 0.02678213620884411,
                                           0.06483020053282014]
diff_compress_encrypt_high_entropy_dist = scale_to(diff_list_to_list(diff_compress_encrypt_high_entropy_dist), 1048576)

raptor_dist = [0, 10241, 491582, 712794, 831695, 831695, 831695, 831695, 831695, 831695, 948446, 1032189, 1032189,
               1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189,
               1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189,
               1032189, 1032189, 1032189, 1032189, 1032189, 1048576]


def encode_and_save(chunk_size=70):
    res = []
    # add all files in ./datasets/ to the list:
    for file in [glob.glob("./datasets/*")[1]]:
        for id_spacing in [0, 4, 8]:
            for use_payload_xor in [True, False]:
                print(f"Running: {file} - {id_spacing} - {use_payload_xor}")
                for dist_name, dist in [("raptor", raptor_dist), ("bmp_low_entropy_evo_dist", bmp_low_entropy_evo_dist),
                                        ("bmp_low_entropy_diff_dist", bmp_low_entropy_diff_dist),
                                        ("evo_compress_encrypt_high_entropy_dist",
                                         evo_compress_encrypt_high_entropy_dist),
                                        ("diff_compress_encrypt_high_entropy_dist",
                                         diff_compress_encrypt_high_entropy_dist)]:
                    res.append({"file": file, "id_spacing": id_spacing, "mask_id": True,
                                "use_payload_xor": use_payload_xor, "dist": dist,
                                "packets": encode(file, chunk_size, dist, return_packets=True, repeats=0,
                                                  id_spacing=id_spacing,
                                                  mask_id=True, use_payload_xor=use_payload_xor,
                                                  insert_header=False,
                                                  seed_struct_str="H", return_packet_error_vals=False,
                                                  store_packets=True)})
    with open("large_scale_experiment.json", "w") as f:
        json.dump(res, f)


def encode_big_buck_bunny():
    file = "big_buck_bunny_1080p_surround.avi"
    id_spacing = 7
    use_payload_xor = True
    dist = evo_compress_encrypt_high_entropy_dist
    chunk_size = 69
    print(f"Running: {file} - {id_spacing} - {use_payload_xor}")
    # for dist_name, dist in [("raptor", raptor_dist), ("bmp_low_entropy_evo_dist", bmp_low_entropy_evo_dist),
    #                        ("bmp_low_entropy_diff_dist", bmp_low_entropy_diff_dist),
    #                        ("evo_compress_encrypt_high_entropy_dist",
    #                         evo_compress_encrypt_high_entropy_dist),
    #                        ("diff_compress_encrypt_high_entropy_dist",
    #                         diff_compress_encrypt_high_entropy_dist)]:
    mp_gen_min_packets(filename="sleeping_beauty", repair_symbols=2, while_count=13511121, out_size=13511121,
                       chunk_size=chunk_size, sequential=True, spare1core=True, insert_header=False,
                       seed_len_format="I", method='RU10', mode1bmp=False, drop_above=1.0,
                       save_as_fasta=True, packets_to_create=13511121, xor_by_seed=use_payload_xor,
                       id_spacing=id_spacing, custom_dist=dist)

def encode_dataset(files, dna_fountain_dir):
    # save the current working dir:
    current_dir = os.getcwd()

    def run_dna_fountain_command(abs_file, filename, dna_fountain_dir):
        if len(glob.glob(f"{current_dir}/datasets/out/ez_{filename}_nc*.fasta")) > 0:
            return
        command = f"cd {dna_fountain_dir.strip()} && " \
                  f"source venv/bin/activate && " \
                  f"python encode.py -f {abs_file} -l 23 -m 3 --gc 0.10 --rs 2 --delta 0.05 --c_dist 0.1 --alpha 0.07 --out {filename}.fasta && " \
                  f"cp {filename}.fasta {current_dir}/datasets/out/ez_{filename}.fasta && " \
                  f"cd {current_dir.strip()}"
        # rename the file to the correct name:

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        print(f"{stdout}")
        print(f"{stderr}")
        chunks = int(stdout.decode('utf-8').split("There are ")[1].split(" input segments")[0])
        # rename the file {current_dir}/datasets/out/ez_{filename}.fasta to {current_dir}/datasets/out/ez_{filename}_nc{chunks}.fasta:
        os.rename(f"{current_dir}/datasets/out/ez_{filename}.fasta",
                  f"{current_dir}/datasets/out/ez_{filename}_nc{chunks}.fasta")
        return stdout.decode('utf-8')

    for file in files:
        # get filename only from path given in file using python libs:
        file_name = os.path.basename(file)
        # get the path:
        path = os.path.dirname(file)
        # get the full path:
        full_path = os.path.abspath(file)
        if len(glob.glob(f"{current_dir}/datasets/out/grass_{file_name}_blocks*.fasta")) == 0:
            # encode using grass code using the external executable "./texttodna --encode --input <file>  --output <file>.dna":
            process = subprocess.Popen(
                f"cd datasets/grass && ./texttodna --encode --input {full_path} --output /tmp/{file_name}.dna && cd {current_dir.strip()}",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, _ = process.communicate()
            blocks = stdout.decode("utf-8").split(" blocks,")[0].split(" ")[-1]
            # open the output file and get the length of the first line (without newline):
            with open(f"/tmp/{file_name}.dna", "r") as f:
                lines = f.readlines()
                grass_length = len(lines[0].strip())  # should be always 117 !
            if grass_length != 117:
                raise ValueError("Grass code produced a sequence with a length != 117!")
            # write to a fasta file:
            with open(f"{current_dir}/datasets/out/grass_{file_name}_blocks{blocks}.fasta", "w") as o_:
                for i, line in enumerate(lines):
                    o_.write(f">grass_{i}\n")
                    o_.write(f"{line}\n")

        # encode using DNA Fountain:
        run_dna_fountain_command(full_path, file_name, dna_fountain_dir)

        # encode using optimized codes:
        for dist_name, dist in {"bmp_low_entropy_evo_dist": bmp_low_entropy_evo_dist,
                                "evo_compress_encrypt_high_entropy_dist": evo_compress_encrypt_high_entropy_dist,
                                "raptor_dist": raptor_dist}.items():
            for rs in [2, 3, 4]:
                if os.path.exists(f"{current_dir}/datasets/out/{file_name}_{dist_name}_rs{rs}_nc{blocks}.fasta"):
                    continue
                encoder = encode_to_fasta(f"{full_path}", 0, get_error_correction_encode("reedsolomon", rs), False,
                                          True, 2, False, in_dist=dist, seed_len_str="I", chunk_size=25 - rs)
                number_of_chunks = encoder.number_of_chunks
                encoder.save_packets_fasta(
                    f"{current_dir}/datasets/out/{file_name}_{dist_name}_rs{rs}_nc{number_of_chunks}.fasta",
                    seed_is_filename=True)
                # save the number of chunks to a file:
                # with open(f"datasets/out/{file}_{dist_name}_number_of_chunks.txt", "w") as o_:
                #    o_.write(f"{number_of_chunks}")


def introduce_errors(folder, mesa_mode=False, mesa_apikey="IgGD6Cfdlnqa4tUungucZpKp3hfYkt1IDqg0Bn3BxEE"):
    # introduce random errors and save them in clusts/error/...:
    # files = glob.glob("clusts/cs_23_I_max_2_hp_10_gc_*.fasta")
    # files.append("clusts/sleeping_beauty_grass.fasta")
    # files.append("clusts/sleeping_beauty_dna_fountain_cs23.fasta")
    # get all grass files:
    files = glob.glob(f"{folder}/grass*.fasta")

    exp_res = ["file,error_rate,repeat,subs,ins,dels,drop"]
    for file in files:
        # get filename only from path given in file using python libs:
        file_name = os.path.basename(file)
        # get the path:
        path = os.path.dirname(file)
        # get the full path:
        full_path = os.path.abspath(file)
        grass_enc = load_fasta(file)
        # calculate the length of the sequences:
        length = "".join(list(grass_enc.values())).replace("\n", "").replace("\r", "").replace(" ", "")
        # get the other matching files for this experiment:
        exp_files = glob.glob(f"{file.replace('grass_', '*').split('_blocks')[0]}*.fasta")

        if mesa_mode:
            for error_multiplier in [0.8, 1.0, 1.02, 1.04, 1.06]:
                for repeat in range(0, 4):
                    for file in exp_files:
                        # load file as a fasta file:
                        total_nts = 0
                        out = {}
                        seqs = load_fasta(file)
                        for key, seq in seqs.items():
                            seq = seq.strip()
                            total_nts += len(seq)
                            out[key] = get_mesa_errors_seqs(seq, error_multiplier, mesa_apikey)
                            if total_nts >= len(length):
                                break

                        # save the sequences to a new file:
                        out_file = f"{folder}/mesa_error/{file_name}_mesaerror_{error_multiplier}.fasta"
                        with open(out_file, "w") as o_:
                            for key, value in out.items():
                                o_.write(f">{key}\n")
                                o_.write(f"{out[key]}\n")
                        exp_res.append(f"{out_file},{error_multiplier},{repeat},mesa,mesa,mesa,mesa")

        for error_rate in [0.01, 0.02, 0.03, 0.04, 0.05]:
            # iterate over all files in "files" and add mutations to the sequences:
            for repeat in range(0, 4):
                for file in exp_files:
                    # load file as a fasta file:
                    total_nts = 0
                    seqs = load_fasta(file)
                    out, (subs, ins, dels, drop) = random_errors(seqs, error_rate, 0.00001, max_nts=len(length))

                    # save the sequences to a new file:
                    out_file = f"{folder}/error/{file_name}_error_{error_rate}.fasta"
                    with open(out_file, "w") as o_:
                        for key, value in out.items():
                            o_.write(f">{key}\n")
                            o_.write(f"{out[key]}\n")
                    exp_res.append(f"{out_file},{error_rate},{repeat},{subs},{ins},{dels},{drop}")
    # save exp_res to a csv file:
    with open(f"{folder}/{'error' if not mesa_mode else 'mesa_error'}/error_results.csv", "w") as o_:
        for line in exp_res:
            o_.write(f"{line}\n")


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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
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


def analyze():
    # open the csv using pandas:

    df = pd.read_csv("clusts/error/error_results.csv")
    # show only rows with success == True:
    df[df["success"] == False]

    ###################################:

    number_of_chunks = 2868
    files = glob.glob("clusts/opt_lorem_ipsum100k.doc_150*.fasta")

    for file in files:
        error_correction = get_error_correction_decode("reedsolomon", 2)
        dist = get_dist(file)
        use_payload_xor = get_payload_xor(file)
        seed_spacing = get_seed_spacing(file)
        static_number_of_chunks = number_of_chunks
        print(
            f"Current file and settings: {file}, {dist}, {use_payload_xor}, {seed_spacing}, {static_number_of_chunks}")
        try:
            res = decode_from_fasta(file, number_of_chunks=number_of_chunks, dist=dist,
                                    error_correction=error_correction,
                                    use_seed_xor=False, use_payload_xor=use_payload_xor, seed_spacing=seed_spacing,
                                    use_headerchunk=False)
        except Exception as e:
            print(f"Error in {file}: {e}")
            continue

    # idea: set of files with different entropy / characteristics: one file per folder, generate fasta file using
    # DNA Fountain, Grass and all optimized codes, then introduce random errors (either MESA or random)
    # while limiting to the max number of bases used by grass. Then tying to decode them using the different decoders.
    # Save the results in a csv file.


if __name__ == "__main__":
    dna_fountain_dir = "/home/schwarz/dna-fountain"  # "/home/schwarz/dna-fountain"
    # bmp_files = glob.glob("datasets/BMP_tiny/001*-bmp.bmp")
    bmp_files = [f"datasets/BMP_tiny/001{i}-bmp.bmp" for i in range(5)]
    # xlsx_files = glob.glob("datasets/XLSX_tiny/002*-xlsx.xlsx")
    xlsx_files = [f"datasets/XLSX_tiny/002{i}-xlsx.xlsx" for i in range(5)]
    # zip_high_files = glob.glob("datasets/ZIP_HIGH_tiny/003*-zip-highcompress.zip")
    zip_high_files = [f"datasets/ZIP_HIGH_tiny/003{i}-zip-highcompress.zip" for i in range(5)]
    # txt_files = glob.glob("datasets/TXT_tiny/004*-txt.txt")
    txt_files = [f"datasets/TXT_tiny/004{i}-txt.txt" for i in range(5)]

    all_files = bmp_files + xlsx_files + zip_high_files + txt_files
    encode_dataset(all_files, dna_fountain_dir)

    # create errors using mesa:
    introduce_errors("/home/schwarz/OFC4DNA/datasets/out", mesa_mode=True,
                     mesa_apikey="grM5qnMhlB-UhSAJQt8wXBb4g85Mj6vJ6qrLudOKNLA")

    # crewate errors simple:
    introduce_errors("/home/schwarz/OFC4DNA/datasets/out", mesa_mode=False)

    try_decode("/home/schwarz/OFC4DNA/datasets/out/mesa_error", dna_fountain_dir)

    try_decode("/home/schwarz/OFC4DNA/datasets/out/error", dna_fountain_dir)
