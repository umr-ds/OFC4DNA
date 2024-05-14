import glob
import os

import numpy as np
from norec4dna import get_error_correction_encode
from norec4dna.rules.FastDNARules import FastDNARules

import Distribution
from Helper import to_dist_list, norm_list, encode


def encode_to_fasta(file_in, chunk_size, rs_symbols, dist, overhead_fac=None, avg_error_fac=None,
                    clean_deg_len_fac=None, clean_avg_error_fac=None, non_unique_packets_fac=None, seed_spacing=None,
                    use_payload_xor=True, seed_struct_str="H"):
    # generate all dna sequences from the file using the distribution function `dist`
    error_correction = get_error_correction_encode("reedsolomon", rs_symbols)
    dist = Distribution.Distribution(norm_list(to_dist_list(dist)), overhead_fac=overhead_fac,
                                     avg_error_fac=avg_error_fac, clean_deg_len_fac=clean_deg_len_fac,
                                     clean_avg_error_fac=clean_avg_error_fac,
                                     non_unique_packets_fac=non_unique_packets_fac,
                                     seed_spacing=seed_spacing, use_payload_xor=use_payload_xor)

    return encode(file_in, chunk_size, dist.to_raptor_list(), repeats=1, return_packets=True,
                  return_packet_error_vals=False, id_spacing=seed_spacing, use_payload_xor=use_payload_xor,
                  error_correction=error_correction, seed_struct_str=seed_struct_str, packets_to_create=65536)


def load_fasta(fasta_file):
    """
    Loads fasta file and returns a dictionary of sequences
    """
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_name = line.strip()[1:]
                fasta_dict[seq_name] = ''
            else:
                fasta_dict[seq_name] += line.strip()
    return fasta_dict


def check_seqs(seqs):
    """
    calculate the error rate for all sequences
    """
    res = {}
    x = FastDNARules()
    for seq in seqs:
        res[seq] = x.apply_all_rules(seq)
    return res


if __name__ == "__main__":
    """
    In DNA-fountain folder:
    - patch the files using the following command:
    git apply <path to patch file>
    - build the extension using the following command:
    python setup.py build_ext --inplace
    - run the following command to encode each file for different chunk sizes (size parameter):
    python encode.py --file_in Dorn --m 1000 --gc 0.99 --size 40 --rs 2 --out dorn_ez.fasta --stop 65536
    - copy the generated fasta files to the eval/ez folder
    """

    aes_dorn_evo_best = [0.0, 0.02088799265672318, 0.023699988666651232, 0.01643490319999933, 0.017412252369953574,
                         0.027759268973297727, 0.028157601781817997, 0.02258613853687647, 0.034596766258260137,
                         0.02887242478245877, 0.026818528623717332, 0.02316381047240955, 0.02529651292374893,
                         0.020982484715106033, 0.030756801808736363, 0.0, 0.02757494612860611, 0.021387981216317905,
                         0.024865855541047515, 0.0, 0.02891732486636941, 0.021864007284190537, 0.02681611820527676,
                         0.02254534410164049, 0.029479746203610668, 0.03455324392837791, 0.02804994960306137,
                         0.028725225100967516, 0.03515417907168796, 0.03361078343361947, 0.029009654487853788,
                         0.030314200195660612, 0.026546443549719655, 0.03402998655066935, 0.030580261587484246,
                         0.035575445677210965, 0.026881705960461674, 0.02603731249299422, 0.02325741010076499,
                         0.026797398942650603
                         ]

    aes_dorn_diff_best = [0.006918697490223755, 0.019780025380762272, 0.02267846770384564, 0.018949709831797563,
                          0.03162725040000313, 0.028372166193997992, 0.01970551392921105, 0.010272036186997055,
                          0.010706960895218436, 0.013248559839070478, 0.040716915888522304, 0.04042797756960299,
                          0.03749710987201956, 0.0195278584236879, 0.030284211292832052, 0.018780709427957218,
                          0.03842539130082528, 0.041340697069709394, 0.03685813521559906, 0.01598522910624237,
                          0.021188427488971064, 0.029030349153101494, 0.02827441474282986, 0.024982859393149766,
                          0.041314007163255485, 0.025225402873826918, 0.04104964161719164, 0.035190273206113926,
                          0.014256329493491163, 0.005164065468535932, 0.0015369733299758481, 0.03772561343752855,
                          0.010087572829176671, 0.042695524183949826, 0.030344773883911994, 0.010256567637385703,
                          0.04164496368083376, 0.03586767853015908, 0.0073614066286906, 0.0146995322397952]

    bmp_low_entropy_evo_best = [
        0.0, 0.0, 0.0, 0.001991538778381074, 0.0, 0.0017167867370232852, 0.0, 0.0, 0.0, 0.05516589457756296, 0.0,
        0.0299235802004363, 0.029847243399513806, 0.0353386889939588, 0.02369075127837102, 0.05049810165140098,
        0.04045382120403153, 0.048821483865876605, 0.029403110724376905, 0.0, 0.0, 0.03984805686387271,
        0.00976099934029729, 0.015388026022486755, 0.02758844640120424, 0.030851991786954046, 0.04676328401692941,
        0.011111961911194148, 0.02288029368703013, 0.059496930480692795, 0.03352347781178974, 0.023214520590854493,
        0.021301650297682287, 0.034349493352951996, 0.03772101728166787, 0.05461358294545733, 0.04183742038736362,
        0.020978552909972935, 0.08063905462190288, 0.041280237878761936]

    bmp_low_entropy_diff_best = [5.227453033901175e-05,
                                 0.006890018998656997, 0.031926045626444, 0.04939393261352515, 0.015120884138625728,
                                 0.0, 0.03114288933188029, 0.05314292058426036, 0.04604504875230231, 0.0651382455768513,
                                 0.014268704956487554, 0.012581814487969302, 0.04385847052067237, 0.050476630364494963,
                                 0.005604681112762536, 0.0, 0.031131860341310245, 0.037959893344564365,
                                 0.05092772889108857, 0.05526470971823938, 0.0012186291699208297, 0.028386269273227627,
                                 0.07061860801032167, 0.012732910606914926, 0.03144468794227678, 0.010238089227963432,
                                 0.004309638335131354, 0.0, 0.0409166864739763, 0.0011472154657503296,
                                 0.040736005281202525, 0.0, 0.028365364661371753, 0.03305288415424515,
                                 0.01524390215313419, 0.0, 0.05492167595159645, 0.0, 0.025740679402491994, 0.0]
    seed_struct_str = "I"
    seed_spacing = 4
    for (name, method) in [("aes_Dorn_evo", aes_dorn_evo_best), ("aes_Dorn_diff", aes_dorn_diff_best),
                           ("bmp_low_entropy_evo", bmp_low_entropy_evo_best), ("bmp_low_entropy_diff",
                                                                               bmp_low_entropy_diff_best)]:
        for chunks in [40, 60, 80, 100]:
            file = "aes_Dorn" if "aes" in name else "sleeping_beauty"
            packets = encode_to_fasta(file, chunks, 0, method, seed_spacing=seed_spacing, use_payload_xor=True,
                                      seed_struct_str=seed_struct_str)
            # fixme:
            unique_packets = set([tuple(packet.data) for packet in packets])
            with open(f"eval/ez/{name}_{seed_struct_str}.fasta{chunks}_{len(packets) - len(unique_packets)}.fasta", 'w') as f:
                for i, packet in enumerate(packets):
                    f.write(f">packet_{i}\n{packet.dna_data}\n")

    # """
    # get all files ending in .fasta in the "eval/ez" folder
    files = glob.glob(os.path.join("eval/ez", "*.fasta"))
    # files = ["/home/michael/Code/dna-fountain/dorn_ez.fasta40_18142.fasta",
    #         "/home/michael/Code/dna-fountain/dorn_ez.fasta60_22018.fasta",
    #         "/home/michael/Code/dna-fountain/dorn_ez.fasta80_24135.fasta",
    #         "/home/michael/Code/dna-fountain/dorn_ez.fasta100_25345.fasta",
    #         "/home/michael/Code/dna-fountain/aes_Dorn_ez.fasta40_18142.fasta",
    #         "/home/michael/Code/dna-fountain/aes_Dorn_ez.fasta60_22018.fasta",
    #         "/home/michael/Code/dna-fountain/aes_Dorn_ez.fasta80_24135.fasta",
    #         "/home/michael/Code/dna-fountain/aes_Dorn_ez.fasta100_25345.fasta"]
    for fasta_file in files:
        fasta_dict = load_fasta(fasta_file)
        seqs = check_seqs(fasta_dict.values())
        # return the number of sequences that have an error rate of < 1.0
        print(f"File without path: {fasta_file.split('/')[-1]}")
        print(f"Total sequences: {len(seqs)}")
        print(f"Chunk size: {fasta_file.split('.fasta')[1].split('_')[0]}")
        print(f"Sequences with error < 1.0: {len([x for x, y in seqs.items() if y < 1.0])}")
        print("Sequences with error >= 1.0: {}".format(len([x for x, y in seqs.items() if y >= 1.0])))
        print(f"Non-unique sequences (only differ in seed): {fasta_file.split('.fasta')[1].split('_')[1]}")
        print(f"Avg error: {sum(seqs.values()) / len(seqs)}")
        # calculate the variance of the error rate using numpy:
        print(f"Variance: {np.var(list(seqs.values()))}")
        print(f"Sequence length: {len(list(fasta_dict.values())[0])}")
        # print(f"(NOREC4DNA) Sequence length with chunk_size = 40 (16bit seed): 168")
        print("-------------------------------------------------------------")

    """
    #"""
