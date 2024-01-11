# Optimizer-Suite for Distribution functions of FountainCodes for non-Gaussian Error-Channels

This tool is intended to be used for optimizing the distribution functions of FountainCodes for non-Gaussian error
channels.
It is based on the NOREC4DNA framework ( https://github.com/umr-ds/NOREC4DNA ) and uses the Differential Evolution,
Evolutionary Optimization and Gradient Descent algorithms to improve the distribution functions for a given (set of)
file(s) under various configurable conditions.

As this tool was created as part of a research project for the field for DNA storage, it is currently optimized for this
use case.

### Requirements

For the installation, Python 3.8 or newer and pip is required (we recommend 22.1.2 or newer)
The software has been tested on Ubuntu 20.04, Ubuntu 22.04 and Windows 11, but should work on any system supporting
Python 3.

### Installation

```bash
git clone <git_url> --recurse-submodules
cd <project folder>

# optional: create a virtual environment
python -m venv venv
source venv/bin/activate

# then run:
pip install -r requirements.txt
```

This will install all required dependencies including the NOREC4DNA framework in the virtual environment.
The installation should take less than 5 minutes on a typical desktop pc, but can take longer if an older pip version is
used or if dependency conflicts exists
(e.g. if no virtual environment is used).

### Usage:

OptimizationSuite.py is a Python script that allows you to configure various optimization parameters and perform
experiments with different settings.
It provides a command-line interface for setting parameters and specifying input files.

```bash
python OptimizationSuite.py [options]
```

#### Options

- --repeats [INT]: Number of repeats (default: 10).
- --chunksize_lst [INT INT INT ...]: List of chunk sizes (default: [40, 60, 80]).
- --overhead_fac [FLOAT]: Overhead factor (default: 0.2).
- --avg_error_fac [FLOAT]: Average error factor (default: 0.4).
- --clean_deg_len_fac [FLOAT]: Clean degree length factor (default: -0.0001).
- --clean_avg_error_fac [FLOAT]: Clean average error factor (default: 0.2).
- --non_unique_packets_fac [FLOAT]: Non-unique packets factor (default: 0.3).
- --unrecovered_packets_fac [FLOAT]: Unrecovered packets factor (default: 0.1).
- --gens [INT]: Number of generations (default: 250).
- --pop_size [INT]: Population size (only for diff and evo) (default: 100).
- --cr [FLOAT]: Crossover rate (for diff) (default: 0.8).
- --f [FLOAT]: Scaling factor (for diff) (default: 0.8).
- --mut_rate [FLOAT]: Mutation rate (for evo) (default: 0.2).
- --alpha [FLOAT]: Alpha (for grd) (default: 0.001).
- --seed_spacing [INT]: Seed spacing (0 is same as no seed-spacing) (default: 4).
- --use_payload_xor [True/False]: Use payload XOR (default: True).
- --compare_pop_size [INT]: Population size for hyperparameter comparison (default: 100).
- --compare_gens [INT]: Number of generations for hyperparameter comparison (default: 100).
- --files [FILE1 FILE2 ...]: One or more input files (default: ["aes_dorn"]).

#### Example

To run the script with custom parameters and multiple input files:

```bash
python OptimizationSuite.py --repeats 5 --avg_error_fac 0.3 --seed_spacing 2 --files Dorn Dorn.pdf aes_Dorn
```

The resulting data will be stored in the matching folders (e.g. `results_evo`)

##### Example output:

The `results` folder contains example data for various input files.

```bash
python OptimizationSuite.py
```

### Evaluation

The results can be parsed and evaluated using the `eval_and_plot.ipynb` Notebook.

Results for various example input files are already included in the `eval` folder.
### Runtime

The runtime depends on various factors, such as the number of repeats, the number of generations, the population size,
the number of input files and the number of chunks.

For Evolutionary Optimization and Differential Evolution, the process can be parallelized by using multiple processes.
This is limited by the population size.

When using multiple input files, chunk sizes, repeats as well as a population size of 100+ and 200+ generations,
for a typical desktop pc, a runtime in the order of multiple days has to be expected.

#### Running on a server using docker:

> mkdir dist_opt_volume
>
> docker build -t dist_opt .
>
> docker run -d --name dist_opt --mount source=dist_opt_volume,target=/optimize/results dist_opt
>
> docker logs -f dist_opt

by default, the container will run the optimization suite with the default parameters.


### Other improvements:
All improvemtns shown in the paper were directly integrated into the NOREC4DNA framework and can either directly or 
using this optimization suite.
For more information, see the NOREC4DNA repository:
[https://github.com/umr-ds/NOREC4DNA](https://github.com/umr-ds/NOREC4DNA)

## Cite

If you use this software in your research, please cite the following paper:

```
TBD
```