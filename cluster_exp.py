import glob
import json
import multiprocessing
import os
import typing
from math import ceil

from NOREC4DNA.find_minimum_packets import main
from NOREC4DNA.optimizer.optimization_helper import diff_list_to_list, scale_to
import RNA

from compare_dna_fountain import load_fasta

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

from norec4dna.rules.FastDNARules import FastDNARules
from norec4dna import RaptorDistribution
from NOREC4DNA.norec4dna import RU10Encoder, get_error_correction_encode
from Helper import norm_list, to_dist_list, encode, scale_to
from Distribution import Distribution

ID_LEN_FORMAT = "H"
NUMBER_OF_CHUNKS_LEN_FORMAT = "I"
CRC_LEN_FORMAT = "I"
PACKET_LEN_FORMAT = "I"
DEFAULT_CHUNK_SIZE = 40
upper_bound = 1.0
error_correction = get_error_correction_encode("reedsolomon", 2)
mask_id = False
overhead = 1


def mesa_encode(file, dist_func, use_payload_xor, seed_spacing):
    # create all packets for the dist:
    # dist = RaptorDistribution(number_of_chunks)
    # dist.f = dist_func
    # dist.d = [x for x in range(0, 41)]
    # encode(file, chunk_size, dist, rules=None, return_packets=False, repeats=5, id_spacing=0, mask_id=True,
    #            use_payload_xor=True, insert_header=False, seed_struct_str="H", return_packet_error_vals=False,
    #            store_packets=True):
    packets = encode(file, 35, dist_func, rules=FastDNARules(), return_packets=True, repeats=0,
                     id_spacing=seed_spacing, mask_id=False, use_payload_xor=use_payload_xor, insert_header=False,
                     return_packet_error_vals=False, store_packets=True)
    return packets


def save_packets_fasta(out_file, packets, seed_is_rowname=True):
    if not out_file.endswith(".fasta"):
        out_file = out_file + ".fasta"
    i = 0
    abs_dir = os.path.split(os.path.abspath("../" + out_file))[0]
    if not os.path.exists(abs_dir):
        os.makedirs(abs_dir)

    with open(out_file, "w") as f:
        for packet in packets:
            if seed_is_rowname:
                i = packet.id
            e_prob = (str(ceil(packet.error_prob * 100)) + "_") if packet.error_prob is not None else ""
            f.write(">" + e_prob + str(i) + "\n" + packet.get_dna_struct(True) + "\n")
            i += 1
    print(f"Saved result at: {out_file}")


def process_file(file, temperature):
    RNA.cvar.temperature = temperature
    print(f"Processing {file}")
    fasta = load_fasta(file)
    tmp = []
    for title, seq in fasta.items():
        rna = RNA.fold_compound(seq)
        pf = rna.pf()[1]
        tmp.append(pf)
        # limit to 5000 to be comparable to the other results:
        if len(tmp) >= 5000:
            break
        # print("PF: ", pf)
    return {file: tmp}


def process_glob():
    files = glob.glob("clusts/sleeping_beauty_grass.fasta")
    res = {}
    cores = multiprocessing.cpu_count() - 1

    with multiprocessing.Pool(cores) as pool:
        res = pool.starmap(process_file, [(file, 37) for file in files])
    # dump as json:
    with open("mfe.json", "w") as f:
        json.dump(res, f)
    return res


def eval_max_free_sec_struct(folder):
    files = glob.glob(folder + "/sleeping_beauty_150*.fasta")
    files.append("clusts/sleeping_beauty_grass.fasta")
    print(files)
    exit(0)
    res = {}
    cores = multiprocessing.cpu_count() - 1

    for file in files:
        with multiprocessing.Pool(cores) as pool:
            res[file] = pool.starmap(process_file, [(file, temperature) for file in files])
    # dump as json:
    with open(folder + "/mfe.json", "w") as f:
        json.dump(res, f)
    return res


def convert_to_fasta():
    # load the file: "./clusts/sleeping_beauty_grass.fasta":
    with open("./clusts/sleeping_beauty_grass.dna", "r") as f:
        lines = f.readlines()

    with open("./clusts/sleeping_beauty_grass.fasta", "w") as f:
        for i, line in enumerate(lines):
            f.write(">" + str(i) + "\n" + line)


def generate_for_mfe(file, distname, dist, use_payload_xor, seed_spacing):
    _, foldername, _ = main(filename=file, repair_symbols=2, while_count=65536, out_size=65536, chunk_size=35,
                            sequential=True, spare1core=True, seed_len_format="H",
                            method='RU10', mode1bmp=False, drop_above=0.4, save_as_fasta=True,
                            packets_to_create=None, save_as_zip=False, xor_by_seed=use_payload_xor,
                            id_spacing=seed_spacing,
                            custom_dist=dist)
    # move the file to ./clusts:
    filename = glob.glob(f"{foldername}/*.fasta")[0]  # fasta file is in a foler...
    os.rename(filename,
              f"./clusts/opt_{file}_150_{distname}_seedspacing{seed_spacing}{'_payloadxor' if use_payload_xor else ''}.fasta")


if __name__ == "__main__":
    # convert_to_fasta()

    for distname, dist in [("raptor", raptor_dist), ("bmp_low_entropy_evo_dist", bmp_low_entropy_evo_dist),
                           ("bmp_low_entropy_diff_dist", bmp_low_entropy_diff_dist),
                           ("evo_compress_encrypt_high_entropy_dist", evo_compress_encrypt_high_entropy_dist),
                           ("diff_compress_encrypt_high_entropy_dist", diff_compress_encrypt_high_entropy_dist)]:
        for use_payload_xor in [True, False]:
            for seed_spacing in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                generate_for_mfe("sleeping_beauty", distname, dist, use_payload_xor, seed_spacing)
    exit(0)

    eval_max_free_sec_struct("clusts")
    process_glob()
    # eval_max_free_sec_struct("clusts")
    # exit(0)

    for file in ["sleeping_beauty"]:  # ["Dorn.zip", "logo_mosla_bw.bmp"]:  # ,
        packets = mesa_encode(file, raptor_dist, False, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_raptor_dist.fasta", packets, True)

        packets = mesa_encode(file, raptor_dist, False, 2)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_raptor_dist_seedspacing2.fasta", packets, True)

        packets = mesa_encode(file, raptor_dist, False, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_raptor_dist_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, raptor_dist, False, 6)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_raptor_dist_seedspacing6.fasta", packets, True)

        packets = mesa_encode(file, raptor_dist, False, 8)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_raptor_dist_seedspacing8.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_evo_dist, False, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_low_dist.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_evo_dist, True, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_low_dist_payloadxor.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_evo_dist, False, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_low_dist_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_evo_dist, True, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_low_dist_payloadxor_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, evo_compress_encrypt_high_entropy_dist, False, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_high_dist.fasta", packets, True)

        packets = mesa_encode(file, evo_compress_encrypt_high_entropy_dist, True, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_high_dist_payloadxor.fasta", packets, True)

        packets = mesa_encode(file, evo_compress_encrypt_high_entropy_dist, False, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_high_dist_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, evo_compress_encrypt_high_entropy_dist, True, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_evo_high_dist_payloadxor_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, diff_compress_encrypt_high_entropy_dist, False, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_high_dist.fasta", packets, True)

        packets = mesa_encode(file, diff_compress_encrypt_high_entropy_dist, True, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_high_dist_payloadxor.fasta", packets, True)

        packets = mesa_encode(file, diff_compress_encrypt_high_entropy_dist, False, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_high_dist_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, diff_compress_encrypt_high_entropy_dist, True, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_high_dist_payloadxor_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_diff_dist, False, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_low_dist.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_diff_dist, True, 0)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_low_dist_payloadxor.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_diff_dist, False, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_low_dist_seedspacing4.fasta", packets, True)

        packets = mesa_encode(file, bmp_low_entropy_diff_dist, True, 4)
        packets = [x for x in packets if x.error_prob < 1.0][:5000]
        save_packets_fasta(f"clusts/{file}_150_diff_low_dist_payloadxor_seedspacing4.fasta", packets, True)
