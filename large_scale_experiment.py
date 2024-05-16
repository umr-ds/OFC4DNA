import glob
import json

from Helper import encode
from NOREC4DNA.find_minimum_packets import main as mp_gen_min_packets
from NOREC4DNA.optimizer.optimization_helper import scale_to, diff_list_to_list

"""
The datasets / files can be obtained from:
- MNIST: http://yann.lecun.com/exdb/mnist/
- Netflix Reviews: https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated
- Youtube Trends:: https://www.kaggle.com/datasets/datasnaek/youtube-new?resource=download
- Big Buck Bunny: https://peach.blender.org/download/
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
    mp_gen_min_packets(filename="datasets/big_buck_bunny_1080p_surround.avi", repair_symbols=2, while_count=13511121, out_size=13511121,
                       chunk_size=chunk_size, sequential=True, spare1core=True, insert_header=False,
                       seed_len_format="I", method='RU10', mode1bmp=False, drop_above=1.0,
                       save_as_fasta=True, packets_to_create=13511121, xor_by_seed=use_payload_xor,
                       id_spacing=id_spacing, custom_dist=dist)


if __name__ == "__main__":
    encode_big_buck_bunny()
    #encode_and_save(40)
