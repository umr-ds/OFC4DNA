import argparse
import json

from NOREC4DNA.norec4dna import RU10Encoder, RU10Decoder, get_error_correction_encode, get_error_correction_decode
from NOREC4DNA.norec4dna.distributions.RaptorDistribution import RaptorDistribution
from NOREC4DNA.norec4dna.helper.RU10Helper import intermediate_symbols
from NOREC4DNA.norec4dna.rules.FastDNARules import FastDNARules
from NOREC4DNA.optimizer.optimization_helper import diff_list_to_list, scale_to

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


def encode_to_fasta(filename, number_of_chunks, error_correction, use_seed_xor, use_payload_xor, seed_spacing,
                    use_headerchunk, in_dist, seed_len_str="I", out_file_prefix="out_file", chunk_size=0, overhead=3.0):
    dist = RaptorDistribution(number_of_chunks)
    dist.f = in_dist
    dist.d = [x for x in range(0, 41)]
    encoder = RU10Encoder(filename, number_of_chunks, dist, insert_header=use_headerchunk, pseudo_decoder=None,
                          rules=FastDNARules(), error_correction=error_correction, packet_len_format="I",
                          crc_len_format="I", chunk_size=chunk_size,
                          number_of_chunks_len_format="", id_len_format=seed_len_str,
                          save_number_of_chunks_in_packet=False,
                          mode_1_bmp=False, prepend="", append="", drop_upper_bound=1.0, keep_all_packets=False,
                          checksum_len_str=None, xor_by_seed=use_payload_xor, mask_id=use_seed_xor,
                          id_spacing=seed_spacing)
    encoder.set_overhead_limit(overhead)
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
    res = decoder.decodeFile(packet_len_format="", crc_len_format="I", number_of_chunks_len_format="",
                             id_len_format=seed_len_str)
    print(f"Success: {res}")
    res_data = decoder.saveDecodedFile(last_chunk_len_format="I", null_is_terminator=False, print_to_output=False,
                                       partial_decoding=True)
    return res, res_data, decoder


if __name__ == '__main__':
    """
    dists = {"low_entropy_evo_dist": bmp_low_entropy_evo_dist.tolist(),
             "low_entropy_diff_dist": bmp_low_entropy_diff_dist.tolist(),
             "high_entropy_evo_dist": evo_compress_encrypt_high_entropy_dist.tolist(),
             "high_entropy_diff_dist": diff_compress_encrypt_high_entropy_dist.tolist(),
             "raptor_dist": raptor_dist}

    json.dump(dists, open("dists.json", "w"))
    """

    dists = json.load(open("dists.json", "r"))

    # argparse:
    parser = argparse.ArgumentParser(description='Encode and decode a file using our improved fountain code-based scheme.')

    parser.add_argument('--encode', action='store_true', help='Encode the file.')
    parser.add_argument('--decode', action='store_true', help='Decode the file.')
    parser.add_argument("--dist", metavar="dist", type=str, required=False, default="high_entropy_diff_dist")
    parser.add_argument("filename", metavar="file", type=str, help="the file / folder to Decode")
    parser.add_argument("--error_correction", metavar="error_correction", type=str, required=False,
                        default="nocode", help="Error Correction Method to use; possible values: \
                                    nocode, crc, reedsolomon, dna_reedsolomon (default=nocode)")
    parser.add_argument("--insert_header", required=False, action="store_true", default=False)
    parser.add_argument("--number_of_chunks", metavar="number_of_chunks", required=False, type=int, default=0)
    parser.add_argument("--repair_symbols", metavar="repair_symbols", type=int, required=False, default=2,
                        help="number of repair symbols for ReedSolomon (default=2)")
    parser.add_argument("--seed_spacing", metavar="seed_spacing", type=int, required=False, default=2,
                        help="spacing between seeds (default=2)")
    parser.add_argument("--use_seed_xor", required=False, action="store_true", default=False)
    parser.add_argument("--use_payload_xor", required=False, action="store_true", default=False)
    parser.add_argument("--seed_len_str", required=False, default="I", help="seed length format (default=I)")
    parser.add_argument("--chunk_size", required=False, type=int, default=0, help="chunk size (default=0)")
    parser.add_argument("--overhead", required=False, type=float, default=3.0, help="overhead (default=3.0)")
    args = parser.parse_args()

    if args.dist not in dists.keys():
        print(f"Invalid dist argument: {args.dist}. Possible values: {dists.keys()}")
        exit(1)
    # Encode
    if args.encode:
        if args.number_of_chunks == 0 and args.chunk_size == 0:
            print("Please provide either the number of chunks or the chunk size.")
            exit(1)
        _error_correction = get_error_correction_encode(args.error_correction, args.repair_symbols)
        encoder = encode_to_fasta(args.filename, args.number_of_chunks, _error_correction, args.use_seed_xor,
                                  args.use_payload_xor, args.seed_spacing, args.insert_header, dists[args.dist],
                                  args.seed_len_str, chunk_size=args.chunk_size, overhead=args.overhead)
        encoder.save_packets_fasta(f"{args.filename}_encoded.fasta")
    elif args.decode:
        # Decode
        if args.number_of_chunks == 0:
            print("Please provide the number of chunks.")
            exit(1)
        _error_correction = get_error_correction_decode(args.error_correction, args.repair_symbols)
        res, res_data, decoder = decode_from_fasta(args.filename, args.number_of_chunks, dists[args.dist],
                                                   _error_correction, args.use_seed_xor, args.use_payload_xor,
                                                   args.seed_spacing, args.insert_header, args.seed_len_str)
