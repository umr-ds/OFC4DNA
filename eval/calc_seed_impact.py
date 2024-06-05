import struct

from NOREC4DNA.norec4dna.helper.bin2Quaternary import string2QUATS


def calc_seed_impact(seed_len_str, max_allowed_homopoylmer_run):
    As = "A" * (max_allowed_homopoylmer_run + 1)
    Cs = "C" * (max_allowed_homopoylmer_run + 1)
    Gs = "G" * (max_allowed_homopoylmer_run + 1)
    Ts = "T" * (max_allowed_homopoylmer_run + 1)
    seed_len = struct.calcsize(seed_len_str)
    print(f"Range: {2**(seed_len*8)}")
    invalid = 0
    for i in range(2**(seed_len*8)):
        # translate to dna:
        byte_arr = struct.pack(f"!{seed_len_str}", i)
        dna = "".join(string2QUATS(byte_arr))
        if As in dna or Cs in dna or Gs in dna or Ts in dna:
            invalid += 1
    return invalid


if __name__ == "__main__":
    print(f"Seed impact (H,3): {calc_seed_impact('H', 3)}")
    print(f"Seed impact (I,3): {calc_seed_impact('I', 3)}")
    print(f"Seed impact (H,2): {calc_seed_impact('H', 2)}")
    print(f"Seed impact (I,2): {calc_seed_impact('I', 2)}")
