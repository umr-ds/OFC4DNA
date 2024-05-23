import multiprocessing
import subprocess

from NOREC4DNA.invivo_window_decoder import load_fasta
import os
import numpy as np
import pandas as pd
import base64
import glob
from NOREC4DNA.norec4dna import get_error_correction_decode, RU10Decoder, RU10Encoder, get_error_correction_encode
from NOREC4DNA.norec4dna.distributions.RaptorDistribution import RaptorDistribution
from cluster_exp import raptor_dist, bmp_low_entropy_evo_dist, bmp_low_entropy_diff_dist, \
    evo_compress_encrypt_high_entropy_dist, diff_compress_encrypt_high_entropy_dist
from NOREC4DNA.norec4dna.helper.RU10Helper import intermediate_symbols
from NOREC4DNA.norec4dna.rules.FastDNARules import FastDNARules
# from NOREC4DNA.find_minimum_packets import main as mp_gen_min_packets
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


def run_dna_fountain_command(current_dir, filename, abs_file, dna_fountain_dir, ):
    if len(glob.glob(f"{current_dir}/datasets/out/ez_{filename}_nc*.fasta")) > 0:
        print(f"[EZ] Skipping {filename} as it already exists")
        return
    command = f"cd {dna_fountain_dir.strip()} && " \
              f"source venv/bin/activate && " \
              f"python encode.py -f {abs_file} -l 23 -m 3 --gc 0.10 --rs 2 --delta 0.05 --c_dist 0.1 --alpha 0.07 --out {filename}.fasta && " \
              f"cp {filename}.fasta {current_dir}/datasets/out/ez_{filename}.fasta && " \
              f"cd {current_dir.strip()}"
    # rename the file to the correct name:

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                               executable="/bin/bash")
    stdout, stderr = process.communicate()
    print(f"{stdout}")
    print(f"{stderr}")
    chunks = int(stderr.decode('utf-8').split("There are ")[1].split(" input segments")[0])
    # rename the file {current_dir}/datasets/out/ez_{filename}.fasta to {current_dir}/datasets/out/ez_{filename}_nc{chunks}.fasta:
    os.rename(f"{current_dir}/datasets/out/ez_{filename}.fasta",
              f"{current_dir}/datasets/out/ez_{filename}_nc{chunks}.fasta")
    return stdout.decode('utf-8')


def run_grass_command(current_dir, file_name, full_path):
    if len(glob.glob(f"{current_dir}/datasets/out/grass_{file_name}_blocks*.fasta")) == 0:
        # encode using grass code using the external executable "./texttodna --encode --input <file>  --output <file>.dna":
        process = subprocess.Popen(
            f"cd datasets/grass && ./texttodna --encode --input {full_path} --output /tmp/{file_name}.dna && cd {current_dir.strip()}",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
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
    else:
        print(f"[Grass] Skipping {file_name} as it already exists")


def run_encode_own(current_dir, file_name, full_path):
    for dist_name, dist in {"bmp_low_entropy_evo_dist": bmp_low_entropy_evo_dist,
                            "evo_compress_encrypt_high_entropy_dist": evo_compress_encrypt_high_entropy_dist,
                            "raptor_dist": raptor_dist}.items():
        for rs in [2, 3, 4]:
            if len(glob.glob(f"{current_dir}/datasets/out/{file_name}_{dist_name}_rs{rs}_nc*.fasta")) > 0:
                print(f"[OWN] Skipping {file_name} for {dist_name} ({rs} as it already exists")
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


def process_file(file):
    current_dir = os.getcwd()
    # get filename only from path given in file using python libs:
    file_name = os.path.basename(file)
    # get the path:
    path = os.path.dirname(file)
    # get the full path:
    full_path = os.path.abspath(file)

    # encode using Grass code:
    run_grass_command(current_dir, file_name, full_path)
    # encode using DNA Fountain:
    run_dna_fountain_command(current_dir, file_name, full_path, dna_fountain_dir)
    # encode using optimized codes:
    run_encode_own(current_dir, file_name, full_path)

    os.chdir(current_dir)


def encode_dataset(files):
    # get cpu count:
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count-1) as pool:
        pool.map(process_file, files)


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
    encode_dataset(all_files)

    # set mesa_apikey to the "apikey" from the ENV variables:
    mesa_apikey = os.getenv("apikey")
    # create errors using mesa:
    introduce_errors("/home/schwarz/OFC4DNA/datasets/out", mesa_mode=True,
                     mesa_apikey=mesa_apikey)

    # crewate errors simple:
    introduce_errors("/home/schwarz/OFC4DNA/datasets/out", mesa_mode=False)

    try_decode("/home/schwarz/OFC4DNA/datasets/out/mesa_error", dna_fountain_dir)

    try_decode("/home/schwarz/OFC4DNA/datasets/out/error", dna_fountain_dir)
