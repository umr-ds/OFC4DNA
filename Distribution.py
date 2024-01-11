import functools
import math
import numpy as np
import Helper


@functools.total_ordering
class Distribution:
    def __init__(self, dist_lst, overhead_fac=0.2, avg_error_fac=0.4, clean_deg_len_fac=-0.0001,
                 clean_avg_error_fac=0.2, non_unique_packets_fac=0.3, unrecovered_packets_fac=0.1, seed_spacing=0,
                 use_payload_xor=False):
        self.dist_lst = dist_lst
        self.id_len_format = "H"
        self.non_unique_packets = None
        self.overhead = None
        self.degree_errs = None
        self.avg_err = None
        self.clean_deg_len = None  # number of packets with error < 1
        self.clean_avg_error = None  # average error of packets with error < 1
        self.calculated_error_value = None
        self.avg_unrecovered = None
        self.overhead_fac = overhead_fac
        self.avg_error_fac = avg_error_fac
        self.clean_deg_len_fac = clean_deg_len_fac
        self.clean_avg_error_fac = clean_avg_error_fac
        self.non_unique_packets_fac = non_unique_packets_fac
        self.unrecovered_packets_fac = unrecovered_packets_fac
        self.seed_spacing = seed_spacing
        self.use_payload_xor = use_payload_xor

    def to_json(self):
        """
        Creates a json representation of the distribution.
        """
        return {"dist_lst": self.dist_lst, "overhead": self.overhead, "degree_errs": self.degree_errs,
                "avg_err": self.avg_err, "clean_deg_len": self.clean_deg_len, "clean_avg_error": self.clean_avg_error,
                "overhead_fac": self.overhead_fac, "avg_error_fac": self.avg_error_fac,
                "clean_deg_len_fac": self.clean_deg_len_fac, "clean_avg_error_fac": self.clean_avg_error_fac,
                "calculated_error_value": self.calculated_error_value,
                "non_unique_packets_fac": self.non_unique_packets_fac, "avg_unrecovered": self.avg_unrecovered,
                "unrecovered_packets_fac": self.unrecovered_packets_fac, "non_unique_packets": self.non_unique_packets,
                "seed_space": self.seed_spacing, "use_payload_xor": self.use_payload_xor}

    @staticmethod
    def from_json(state):
        """
        Creates a distribution from a json representation.
        """
        if state is None:
            return None
        dist_lst = state.get("dist_lst")
        tmp = Distribution(dist_lst)
        tmp.overhead = state.get("overhead")
        tmp.non_unique_packets = state.get("non_unique_packets")
        tmp.degree_errs = state.get("degree_errs")
        tmp.avg_err = state.get("avg_err")
        tmp.clean_deg_len = state.get("clean_deg_len")
        tmp.clean_avg_error = state.get("clean_avg_error")
        tmp.overhead_fac = state.get("overhead_fac")
        tmp.avg_error_fac = state.get("avg_error_fac")
        tmp.clean_deg_len_fac = state.get("clean_deg_len_fac")
        tmp.clean_avg_error_fac = state.get("clean_avg_error_fac")
        tmp.non_unique_packets_fac = state.get("non_unique_packets_fac")
        tmp.unrecovered_packets_fac = state.get("unrecovered_packets_fac")
        tmp.avg_unrecovered = state.get("avg_unrecovered")
        tmp.seed_spacing = state.get("seed_space")
        tmp.use_payload_xor = state.get("use_payload_xor")
        calculated_error_value = state.get("calculated_error_value")
        if calculated_error_value != "None" and calculated_error_value is not None:
            tmp.calculated_error_value = calculated_error_value
        return tmp

    def to_raptor_list(self):
        """
        Converts the distribution list to the format used for the RaptorDistribution
        :return:
        """
        lst = np.zeros(len(self.dist_lst) + 1)
        for i in range(1, len(self.dist_lst) + 1):
            lst[i] = self.dist_lst[i - 1] + lst[i - 1]
        return Helper.scale_to(lst, 1048576)

    def compute_fitness(self, file_lst, repeats, chunksize):
        """
        Computes the distribution fitness for all given files with the selected parameters.
        :param file_lst: Files to use.
        :param chunksize: Chunksize to use.
        :return:
        """
        tmp = Helper.compute_distribution_fitness(self.to_raptor_list(), file_lst, repeats, chunksize=chunksize,
                                                  id_len_format=self.id_len_format, seed_spacing=self.seed_spacing,
                                                  use_payload_xor=self.use_payload_xor)
        self.overhead, self.degree_errs, self.clean_avg_error, self.clean_deg_len, self.non_unique_packets, self.avg_unrecovered = tmp
        cnt = 0
        err = 0
        for x in range(0, len(self.degree_errs)):
            if not (math.isclose(self.degree_errs[x], 0.0, rel_tol=1e-09, abs_tol=1e-09) or math.isclose(
                    self.dist_lst[x], 0.0, rel_tol=1e-09, abs_tol=1e-09)):
                cnt += 1
                err += min(1.0, self.degree_errs[x])
        self.avg_err = err / cnt
        return self.avg_err, self.overhead, self.clean_avg_error, self.clean_deg_len, self.non_unique_packets, self.avg_unrecovered

    def save_to_txt(self, name="dist"):
        """
        Creates a txt file with the grade probabilities, the average error and the overhead.
        :param name: Filename to save the distribution with.
        """
        with open(name + "_err_" + str(round(self.avg_err, 2)) + ".txt", "w") as f:
            f.write("Distribution: " + str(self.dist_lst))
            f.write("\nAverage Error: " + str(self.avg_err))
            f.write("\nOverhead: " + str(self.overhead))
            f.write("\nPackets with error smaller 1: " + str(self.clean_deg_len))
            f.write("\nAverage error for packets with error < 1: " + str(self.clean_avg_error))
            f.write("\nCalculated error value: " + str(self.calculated_error_value))
            f.write("\nNon unique packets: " + str(self.non_unique_packets))
            f.write("\nUnrecovered packets: " + str(self.avg_unrecovered))

    def calculate_error_value(self):
        """
        Calculates the error value for the distribution.
        """
        if self.calculated_error_value is None:
            self.calculated_error_value = 100.0 + (self.overhead_fac * self.overhead) + (
                    self.avg_error_fac * self.avg_err) + (self.clean_deg_len_fac * self.clean_deg_len) + (
                                                  self.clean_avg_error_fac * self.clean_avg_error) + (
                                                  self.non_unique_packets_fac * self.non_unique_packets) + (
                                                  self.unrecovered_packets_fac * self.avg_unrecovered)
        return self.calculated_error_value

    def __lt__(self, other):
        if self.overhead is None or self.avg_err is None or \
                self.clean_deg_len is None or self.clean_avg_error is None:
            return False
        return self.calculate_error_value() < other.calculate_error_value()

    def __eq__(self, other):
        return other is not None and self.calculate_error_value() == other.calculate_error_value()
