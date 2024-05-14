import bisect
import math
import multiprocessing
import random
import struct
import typing

import csv
from itertools import chain

import numpy as np
import pandas as pd
import Distribution
from norec4dna.ErrorCorrection import reed_solomon_encode
from norec4dna.Packet import ParallelPacket
from norec4dna.helper.bin2Quaternary import string2QUATS
from norec4dna import RU10Encoder, RU10Decoder, nocode, Encoder
from norec4dna.distributions.RaptorDistribution import RaptorDistribution
from norec4dna.rules.FastDNARules import FastDNARules
from norec4dna.helper.RU10Helper import intermediate_symbols
from norec4dna.helper import should_drop_packet
from norec4dna.RU10Packet import RU10Packet

RU10Encoder.SHOW_PROGRESS_BAR = False


def init_pop(pop_size, add_raptor=False, overhead_fac=0.2, avg_error_fac=0.4, clean_deg_len_fac=-0.0001,
             clean_avg_error_fac=0.2, non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, seed_spacing=0,
             use_payload_xor=False):
    """
    Initializes a randomized population of the given size. If add_raptor is set to True, the default raptor distribution
    will be added to the population.
    :param pop_size: Size of the population to create.
    :param add_raptor: If true, adds the Raptor distribution.
    :return:
    """
    pop = []
    if add_raptor:
        pop.append(Distribution.Distribution(norm_list(to_dist_list(raptor_dist)), overhead_fac, avg_error_fac,
                                             clean_deg_len_fac, clean_avg_error_fac, non_unique_packets_fac,
                                             unrecovered_packets_fac, seed_spacing=seed_spacing,
                                             use_payload_xor=use_payload_xor))
        pop_size -= 1
    for _ in range(0, pop_size):
        pop.append(Distribution.Distribution(mutate_random([1 / 40 for _ in range(0, 40)], 1.0), overhead_fac,
                                             avg_error_fac, clean_deg_len_fac, clean_avg_error_fac,
                                             non_unique_packets_fac, unrecovered_packets_fac, seed_spacing=seed_spacing,
                                             use_payload_xor=use_payload_xor))
    return pop


def mutate_random(dist_lst, fac=0.5):
    """
    Multiplies every probability of the distribution with a random number between 1-fac and 1+fac. Needs distribution
    lists, not Raptor lists.
    :param dist_lst: Distribution list to mutate.
    :param fac: Maximum mutation factor.
    :return:
    """
    mut = np.random.uniform(low=1.0 - fac, high=1.0 + fac, size=40)
    x_new = [x[0] * x[1] for x in zip(dist_lst, mut)]
    return norm_list(x_new)


# //---// #

def norm_list(dist_lst):
    """
    Normalizes distribution lists to 1 in sum.
    :param dist_lst: Distribution list to normalize.
    :return:
    """
    b = sum(dist_lst)
    return [a / b for a in dist_lst]


def shift_to_positive(dist_lst):
    """
    Shifts the distribution list to positive values.
    :param dist_lst: Distribution list to shift.
    :return:
    """
    min_val = min(dist_lst)
    if min_val < 0:
        dist_lst = [x - min_val for x in dist_lst]
    return dist_lst


def to_raptor_list(dist_lst):
    """
    Converts the distribution list to the format used for RaptorDistribution.
    :param dist_lst: Distribution list to convert.
    :return:
    """
    raptor_lst = np.zeros(len(dist_lst) + 1)
    for i in range(1, len(dist_lst) + 1):
        raptor_lst[i] = dist_lst[i - 1] + raptor_lst[i - 1]
    return raptor_lst


def scale_to(x, max_num):
    """
    Scales values of a list to the given max number.
    :param x: List to scale.
    :param max_num: Number to scale to.
    :return:
    """
    x = x / (np.max(x) if np.max(x) != 0 else 1.0)
    return x * max_num


def parallel_to_normal(par_packet, error_correction, dist):
    """
    Converts parallel packets either to RU10Packets, normal Packets or OnlinePackets based on the original class of
    the packet.
    :param par_packet:
    :param error_correction:
    :param dist:
    :return: Converted packet
    """
    if par_packet.get_org_class() == "RU10Packet":
        packet = RU10Packet(data=par_packet.data, used_packets=par_packet.used_packets,
                            total_number_of_chunks=par_packet.total_number_of_chunks, id=par_packet.id, dist=dist,
                            error_correction=error_correction, packet_len_format=par_packet.packet_len_format,
                            crc_len_format=par_packet.crc_len_format,
                            number_of_chunks_len_format=par_packet.number_of_chunks_len_format,
                            id_len_format=par_packet.id_len_format,
                            save_number_of_chunks_in_packet=par_packet.safe_number_of_chunks_in_packet)
        packet.error_prob = par_packet.error_prob
    else:
        raise RuntimeError("Unsupported packet type")
    return packet


def to_dist_list(raptor_lst):
    """
    Converts the raptor list to the format used for the Distribution class.
    :param raptor_lst: Raptor list to convert.
    :return:
    """
    dist_lst = np.zeros(len(raptor_lst) - 1)
    for i in range(1, len(raptor_lst)):
        dist_lst[i - 1] = raptor_lst[i] - raptor_lst[i - 1]
    return dist_lst


def compute_distribution_fitness(raptor_lst, file_lst, runs=25, chunksize=50, id_len_format="H", seed_spacing=0,
                                 use_payload_xor=False):
    """
    Computes the fitness of a distribution with a given fitness function for every given file.
    :param raptor_lst: Raptor list of the distribution to compute the fitness for.
    :param file_lst: List of files to encode with the distribution.
    :param runs: Number of En-/Decoding cycles.
    :param chunksize: Chunksize to use.
    :return:
    """
    overhead_lst = list()
    degree_lst = list()
    non_unique_packets_lst = list()
    avg_unrecovered_lst = list()
    clean_avg_error = 0
    clean_deg_len = 0
    if not isinstance(chunksize, list):
        chunksizelist = [chunksize]
    else:
        chunksizelist = chunksize
    max_unique = int(math.pow(2, 8 * struct.calcsize(id_len_format)))
    for chunksize in chunksizelist:
        for file in file_lst:
            res = encode(file, chunksize, raptor_lst, repeats=runs, return_packets=False,
                         return_packet_error_vals=False, id_spacing=seed_spacing, use_payload_xor=use_payload_xor)[0]
            non_unique_packets_lst.append(max_unique - res[2])
            avg_unrecovered_lst.append(res[3])
            overhead_lst.append(res[0])
            tmp_degree_lst = np.zeros(41)
            for deg in res[1].keys():
                tmp_degree_lst[deg] = min(1.0, sum(res[1][deg]) / len(res[1][deg]))
                tmp = [x for x in res[1][deg] if x < 1.0]
                clean_avg_error += sum(tmp)
                clean_deg_len += len(tmp)
            degree_lst.append(tmp_degree_lst)
    avg_overhead = sum(overhead_lst) / len(overhead_lst)
    avg_unrecovered = sum(avg_unrecovered_lst) / len(avg_unrecovered_lst)
    avg_non_unique_packets = sum(non_unique_packets_lst) / len(non_unique_packets_lst)
    avg_degree_err = []
    active_degs = 0
    for x in range(0, 41):
        tmp_err = 0
        cnt = 0
        for lst in degree_lst:
            if lst[x] != 0:
                scaler = math.log10(41 + 1 - x)  # increase the error for lower degrees
                tmp_err += lst[x] * scaler
                cnt += 1
        if cnt != 0:
            avg_degree_err.append(tmp_err / cnt)
            active_degs += 1
        else:
            avg_degree_err.append(0.0)
    avg_degree_err.pop(0)
    # set all degree errors of degrees with a prob of 0 to the average of the other degrees:
    # this will make the fitness function more stable (less hopping between used and unused degrees)
    avg_without_zero = sum(avg_degree_err) / active_degs
    for i in range(len(avg_degree_err)):
        if avg_degree_err[i] == 0:
            avg_degree_err[i] = avg_without_zero
    clean_avg_error = 1.0 * clean_avg_error / clean_deg_len

    """
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    # Prints out a summary of the large objects
    summary.print_(sum1)
    # Get references to certain types of objects such as dataframe
    dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
    for d in dataframes:
        print(d.columns.values)
        print(len(d))
    """
    return avg_overhead, avg_degree_err, clean_avg_error, clean_deg_len, avg_non_unique_packets, avg_unrecovered


def encode(file, chunk_size, dist, rules=None, return_packets=False, repeats=5, id_spacing=0, mask_id=True,
           use_payload_xor=True, insert_header=False, seed_struct_str="H", return_packet_error_vals=False,
           store_packets=True, error_correction=None, packets_to_create=None):
    """
    Encodes the file into all possible packets, calculate error probability and  the pseudo decoder was able to decode it 'repeats' times with the given chunk size
    and the distribution list.
    :param file: File to encode.
    :param chunk_size: Chunksize to use.
    :param dist: The distribution to calculate the average error and overhead for.
    :param rules: Custom rule-set; defaults to "FastDNARules" if rules is None. Default: FastDNARules.
    :param return_packets: If True, returns the packets instead of the calculated values. Default: False.
    :param repeats: Number of repeats to perform. Default: 5.
    :param id_spacing: Spacing of the seed / id to interleave with the payload. Default: 0.
    :param mask_id: If True, masks the id by applying a fixed mask using XOR. Default: True.
    :param use_payload_xor: If True, uses the payload-xor function. Default: True.
    :param insert_header: If True, inserts a header into the packets. Default: False.
    :param seed_struct_str: Struct string to use for the seed. Default: "H" (unsigned short).
    :param return_packet_error_vals: If True, returns the error values for every packet. Default: False.
    :param store_packets: If True, stores the full packets. Default: True.
    :param error_correction: Error correction method to use (valid: nocode, reed_solomon_encode, ...). Default: nocode.
    :return:
    """
    if error_correction is None:
        error_correction = nocode
    degree_dict = {}
    overhead_lst = []
    unrecovered_lst = []
    number_of_chunks = Encoder.get_number_of_chunks_for_file_with_chunk_size(file, chunk_size, insert_header=False)
    distribution = RaptorDistribution(number_of_chunks)
    distribution.f = dist
    distribution.d = [x for x in range(0, 41)]
    if rules is None:
        rules = FastDNARules()
    encoder = RU10Encoder(file, number_of_chunks, distribution, insert_header=insert_header, rules=rules,
                          error_correction=error_correction, id_len_format=seed_struct_str,
                          number_of_chunks_len_format="B",
                          save_number_of_chunks_in_packet=False, mode_1_bmp=False, xor_by_seed=use_payload_xor,
                          mask_id=mask_id, id_spacing=id_spacing)
    encoder.prepare()
    encoder.random_state = np.random.RandomState()

    # create all possible packets:
    if packets_to_create is None:
        packets_to_create = int(math.pow(2, 8 * struct.calcsize(encoder.id_len_format)))
    packets = []
    packet_error_vals = []
    for i in range(packets_to_create):
        if i % 100000 == 0:
            print("Creating packet %d / %d" % (i, packets_to_create))
        created_packet = encoder.create_new_packet(seed=i)
        should_drop_packet(rules, created_packet)
        if not (return_packets or return_packet_error_vals):
            if created_packet.get_degree() not in degree_dict:
                degree_dict[created_packet.get_degree()] = list()
            degree_dict[created_packet.get_degree()].append(min(created_packet.error_prob, 1.0))
        if store_packets:
            packets.append(created_packet)
        if return_packet_error_vals:
            packet_error_vals.append(created_packet.error_prob)
        # if len([x for x in packet_error_vals if x < 1.0]) > 5:
        #    break
    # if store_packets:
    #    packets = sorted(packets)
    if return_packet_error_vals:
        num_chunks = encoder.number_of_chunks
        del encoder, degree_dict, packets
        return packet_error_vals, num_chunks
    if return_packets:
        del encoder
        return packets
    # insert into pseudo-decoder (ordered by error-probability)
    pseudo_decoder = create_pseudo_decoder(encoder.number_of_chunks, distribution)
    needed_packets = 0
    for _ in range(repeats):
        random.shuffle(packets)
        for packet in packets:
            if pseudo_decoder.GEPP is None or not pseudo_decoder.is_decoded():
                needed_packets += 1
                pseudo_decoder.input_new_packet(packet)
                if needed_packets == encoder.number_of_chunks:
                    unrecovered_lst.append(encoder.number_of_chunks - len(
                        [x for x in pseudo_decoder.GEPP.result_mapping if x != -1]))
            else:
                overhead = (needed_packets - encoder.number_of_chunks) / 100.0
                overhead_lst.append(overhead)
                needed_packets = 0
                pseudo_decoder = create_pseudo_decoder(encoder.number_of_chunks, distribution)
            break
    number_of_chunks = encoder.number_of_chunks
    del encoder, pseudo_decoder

    unrecovered_avg = sum(unrecovered_lst) / len(unrecovered_lst)
    overhead_avg = sum(overhead_lst) / len(overhead_lst)
    return (overhead_avg, degree_dict, len(set([x.get_data().tobytes() for x in packets])), unrecovered_avg), (
        packet_error_vals, number_of_chunks)


def worker(args):
    start_seed, end_seed, file, chunk_size, dist, id_spacing, mask_id, use_payload_xor, insert_header, seed_struct_str = args
    error_correction = nocode
    number_of_chunks = Encoder.get_number_of_chunks_for_file_with_chunk_size(file, chunk_size, insert_header=False)
    distribution = RaptorDistribution(number_of_chunks)
    distribution.f = dist
    distribution.d = [x for x in range(0, 41)]
    rules = FastDNARules()
    encoder = RU10Encoder(file, number_of_chunks, distribution, insert_header=insert_header, rules=rules,
                          error_correction=error_correction, id_len_format=seed_struct_str,
                          number_of_chunks_len_format="B",
                          save_number_of_chunks_in_packet=False, mode_1_bmp=False, xor_by_seed=use_payload_xor,
                          mask_id=mask_id, id_spacing=id_spacing)
    encoder.prepare()
    encoder.random_state = np.random.RandomState()
    packet_error_vals = []
    for i in np.arange(start_seed, end_seed):
        # if i % 100000 == 0:
        #    print(f"Creating packet {i} in ({start_seed}-{end_seed}")
        created_packet = encoder.create_new_packet(seed=i)
        should_drop_packet(rules, created_packet)
        packet_error_vals.append(created_packet.error_prob)
    del encoder, rules
    return packet_error_vals


def encode_for_spacing(file, chunk_size, dist, id_spacing=0, mask_id=True,
                       use_payload_xor=True, insert_header=False, seed_struct_str="H"):
    # create all possible packets:
    total_packets_to_create = int(math.pow(2, 8 * struct.calcsize(seed_struct_str)))
    # use multiprocessing pool to split the work between all cores:
    # set number of workers using the number of cores:
    num_worker = multiprocessing.cpu_count() - 5
    # calculate the number of packets to create per worker using total_packets_to_create and n as the number of workers:
    range_per_worker = total_packets_to_create // num_worker
    with multiprocessing.Pool(num_worker) as p:
        res = p.map(worker, [
            (x, min(x + range_per_worker, total_packets_to_create), file, chunk_size, dist, id_spacing, mask_id,
             use_payload_xor, insert_header, seed_struct_str) for x in
            range(0, total_packets_to_create, range_per_worker)])
        # res = [
        #    worker((x, min(x + range_per_worker, total_packets_to_create), file, chunk_size, dist, id_spacing, mask_id,
        #            use_payload_xor, insert_header, seed_struct_str)) for x in
        #    range(0, total_packets_to_create, range_per_worker)]
    res = list(chain.from_iterable(res))
    return res


def create_pseudo_decoder(number_of_chunks, distribution):
    """
    Creates a new pseudo decoder for the encoder used to encode the files.
    :param number_of_chunks:
    :param distribution: Distribution used for the encoder.
    :return:
    """
    pseudo_decoder = RU10Decoder.pseudo_decoder(number_of_chunks, False)
    if pseudo_decoder.distribution is None:
        pseudo_decoder.distribution = distribution
        pseudo_decoder.numberOfChunks = number_of_chunks
        _, pseudo_decoder.s, pseudo_decoder.h = intermediate_symbols(number_of_chunks, pseudo_decoder.distribution)
        pseudo_decoder.createAuxBlocks()
    return pseudo_decoder


def save_to_csv(file_con, name):
    """
    Saves the given content to a .csv file.
    :param file_con: List of lines to write as csv.
    :param name: Name to save the file with.
    :return:
    """
    with open(name + ".csv", mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for row in file_con:
            csv_writer.writerow(row)


def select_distributions(pop, no_dists):
    """
    Selects the best distributions of the given population based on the average error and the overhead, if overhead_fac
    is greater than 0.
    :param pop: Population to select distributions from.
    :param no_dists: Number of distributions to select.
    :return:
    """
    sorted_dists = sorted(pop, key=lambda x: x.calculate_error_value())
    selected_dists = []
    for y in range(0, no_dists):
        selected_dists.append(sorted_dists[y])
    return selected_dists


def compute_generation(sorted_dists, pop_size, mut_rate=0.05, overhead_fac=0.0):
    """
    Computes the next generation of distributions for the evolutionary optimization.
    The probability for a distribution getting selected as parent is proportional to its cost.
    :param sorted_dists: The population sorted by the distributions costs.
    :param pop_size: The size of the population to compute.
    :param mut_rate: Mutation rate to mutate the created distributions.
    :param overhead_fac: Weight of the overhead.
    :return:
    """
    selected_dists = sorted_dists[:int(pop_size)]
    next_gen = []
    prob_lst = norm_list([x.avg_err + (x.overhead * overhead_fac) for x in selected_dists])
    ind_lst = [y for y in range(0, len(selected_dists))]
    for _ in range(0, pop_size):
        dist_ind = np.random.choice(ind_lst, size=2, p=prob_lst)
        dist_1 = selected_dists[dist_ind[0]]
        dist_2 = selected_dists[dist_ind[1]]
        next_gen.append(merge_dists(dist_1, dist_2, mut_rate))
    return next_gen


def merge_dists(dist_1, dist_2, mut_rate=0.1):
    """
    Merges distributions. Takes the degree probability with the lowest error probability for every degree to create a
    new distribution. Used for the evolutionary optimization.
    :param dist_1:
    :param dist_2:
    :param mut_rate: Mutation rate to mutate the created distribution.
    :return:
    """
    dist_lst = [0] * len(dist_1.dist_lst)
    zer_ents = []
    for i in range(0, len(dist_1.dist_lst)):
        if dist_1.degree_errs[i] < dist_2.degree_errs[i]:
            min_prob = dist_1.dist_lst[i]
        else:
            min_prob = dist_2.dist_lst[i]
        if min_prob < 0.001:
            zer_ents.append(i)
            min_prob = 0
        dist_lst[i] = min_prob
    n_zer_cnt = len(dist_1.dist_lst) - len(zer_ents)
    if n_zer_cnt < 4:
        ind = np.random.choice(zer_ents, size=4 - n_zer_cnt)
        avg_prob = sum([x if x > 0.001 else 0 for x in dist_lst]) / n_zer_cnt
        for j in ind:
            dist_lst[j] = avg_prob
    return Distribution.Distribution(mutate_random(dist_lst, mut_rate), overhead_fac=dist_1.overhead_fac,
                                     avg_error_fac=dist_1.avg_error_fac,
                                     clean_deg_len_fac=dist_1.clean_deg_len_fac,
                                     non_unique_packets_fac=dist_1.non_unique_packets_fac,
                                     unrecovered_packets_fac=dist_1.unrecovered_packets_fac,
                                     seed_spacing=dist_1.seed_spacing, use_payload_xor=dist_1.use_payload_xor)


def get_err_dist(_method, _number_of_chunks, _repair_symbols, dist):
    if _method == 'RU10':
        distribution = RaptorDistribution(_number_of_chunks)
        distribution.f = dist
        distribution.d = [x for x in range(0, 41)]
    else:
        raise NotImplementedError("Only RU10 supported for this mode.")
    return distribution, lambda x: reed_solomon_encode(x, _repair_symbols)


def create_all_packets(in_file, in_dist, number_of_chunks=0, chunk_size=0, seed_len_format="H", repair_symbols=0,
                       method="RU10", drop_above=1.0, l_size=None):
    if chunk_size != 0:
        number_of_chunks = Encoder.get_number_of_chunks_for_file_with_chunk_size(in_file, chunk_size)
    if number_of_chunks == 0:
        raise RuntimeError("Number of Chunks or Chunk Size MUST be != 0 !")
    rules = FastDNARules()
    packets_to_create = int(math.pow(2, 8 * struct.calcsize(seed_len_format)))
    if l_size is None:
        l_size = packets_to_create + 1
    if repair_symbols != 0:
        dist, error_correction = get_err_dist("RU10", number_of_chunks, repair_symbols, in_dist)
    else:
        dist = RaptorDistribution(number_of_chunks)
        dist.f = in_dist
        dist.d = [x for x in range(0, 41)]
        error_correction = reed_solomon_encode
    if method == 'RU10':
        x = RU10Encoder(in_file, number_of_chunks, dist, chunk_size=chunk_size, insert_header=False, rules=rules,
                        error_correction=error_correction, id_len_format=seed_len_format,
                        number_of_chunks_len_format="H",
                        save_number_of_chunks_in_packet=False, mode_1_bmp=False,
                        prepend="", append="")
        x.prepare()
    else:
        raise NotImplementedError("Choose: RU10, LT or Online")
    tmp_list = []
    for i in range(packets_to_create):
        packet = x.create_new_packet(seed=i)
        _ = should_drop_packet(rules, packet)
        if packet.error_prob <= drop_above and (len(tmp_list) < l_size or packet.error_prob < tmp_list[-1].error_prob):
            if packet not in tmp_list:
                bisect.insort_left(tmp_list, packet)
            else:
                elem = next((x for x in tmp_list if x == packet), None)
                if packet < elem:
                    tmp_list.remove(elem)
                    del elem
                    bisect.insort_left(tmp_list, packet)
            if len(tmp_list) > l_size:
                for ele1m in tmp_list[l_size + 1:]:
                    del ele1m
                tmp_list = tmp_list[:l_size]

        else:
            del packet
        i += 1
    del x, rules
    return [ParallelPacket.from_packet(p) for p in tmp_list]


def calculate_entropy(in_file, in_arr=None, convert_to_dna=True):
    """ Calculates the shannon entropy for a given file
    :in_file: input file to calculate the entropy for.
    :in_arr: if set, this function will ignore in_file and work with the provided array
    :convert_to_dna: if set, the input will be converted to DNA-bases.
    :returns: the entropy of the file
    """
    if in_arr is None:
        with open(in_file, 'rb') as f:
            byte_arr = list(f.read())
    else:
        byte_arr = in_arr
    if convert_to_dna:
        byte_arr = list("".join(string2QUATS(byte_arr)))
    file_len = len(byte_arr)
    df = pd.DataFrame(byte_arr)
    df = df.groupby(df.columns[0])[df.columns[-1]].count() / file_len
    return - sum([p * math.log(p, 2) for p in df if p > 0]), df.to_dict()


def calculate_entropys(input_arr, convert_to_dna=True):
    return calculate_entropy(None, in_arr=input_arr, convert_to_dna=convert_to_dna)


raptor_dist = [0, 10241, 491582, 712794, 831695, 831695, 831695, 831695, 831695, 831695, 948446, 1032189, 1032189,
               1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189,
               1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189, 1032189,
               1032189, 1032189, 1032189, 1032189, 1032189, 1048576]


def calculate_unique_packets(file, chunk_size, dist):
    unique_mappings = set()
    packets: typing.List[RU10Packet] = encode(file, chunk_size, dist, None, True)
    number_of_chunks = packets[0].total_number_of_chunks
    distribution = RaptorDistribution(number_of_chunks)
    distribution.f = dist
    distribution.d = [x for x in range(0, 41)]
    pseudo_decoder = create_pseudo_decoder(number_of_chunks, distribution)
    hits = 0
    for packet in packets:
        clean_chunk_lst = "".join(
            [str(1 if x is True else 0) for x in pseudo_decoder.removeAndXorAuxPackets(packet)])
        if clean_chunk_lst in unique_mappings:
            hits += 1
            # print("Hit!")
        else:
            unique_mappings.add(clean_chunk_lst)
    return unique_mappings, hits, number_of_chunks


def create_norec_dist(dist, number_of_chunks):
    if isinstance(dist, Distribution.Distribution):
        dist = dist.dist_lst
    distribution = RaptorDistribution(number_of_chunks)
    distribution.f = dist
    distribution.d = [x for x in range(0, 41)]
    return distribution


if __name__ == '__main__':
    for f in ["Dorn", "LICENSE", "Rapunzel", "Sneewittchen", "Rothkäppchen", "Dorn.pdf", "Dorn.zip",
              "logo_mosla_bw.bmp", "Marburg_0192_Elektroll_CC0.png", "logo_mosla_rgb.png", "logo.jpg",
              "lorem_ipsum100k.doc", "aes_Dorn", "aes_ecb_Dorn"]:
        print(f"---\nFile: {f}\nDNA: {calculate_entropy(f)},\nBytes: {calculate_entropy(f, convert_to_dna=False)[0]}")

    print(calculate_entropy(None, list("ACGT"), convert_to_dna=False))

    id_len_format = "H"
    packets_to_create = int(math.pow(2, 8 * struct.calcsize(id_len_format)))
    dist = raptor_dist
    out = []
    for f in ["Dorn", "logo.jpg", "LICENSE"]:
        for i in range(40, 111, 10):
            res, hits, number_of_chunks = calculate_unique_packets("Dorn", i, dist)
            print(f"File: {f} - chunk size: {i} - #chunks: {number_of_chunks}")
            # print(res)
            print(f"Possible packets: {packets_to_create}")
            print(f"Packets with unique chunks: {len(res)}")
            print(f"Duplicates: {hits}")
            out.append(
                {"res": frozenset(res), "hits": hits, "poss_packets": packets_to_create, "unique_packets": len(res)})
