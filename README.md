# AlphaMapleSAT-CnC

<!-- virtualenv --no-download ams_env
source ams_env/bin/activate
cd alphamaplesat
pip install --upgrade pip
pip install -r requirements_cc.txt

python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 1 -m 136 -o "test.cubes" -prod -->

AlphaMapleSAT is a novel Monte Carlo Tree Search (MCTS) based Cube-and-Conquer (CnC) SAT solving method aimed at efficiently solving challenging combinatorial problems. 

## Features

- **Parallel Execution**: Utilizes multiprocessing for efficient cubing and solving.
- **Flexible Configuration**: Supports multiple solving modes (`satcas`, `exhaustive-no-cas`, `sms`, `smsd2`, `other`) and cubing modes (`march`, `ams`).
- **Customizable Parameters**: Allows fine-tuning of cubing depth, variable cutoff, and solving strategies.
- **Pipeline Integration**: Combines cubing, simplification, and solving into a seamless workflow.

## Installation

1. Clone this repository

2. Run the dependency setup script:
```bash
./dependency-setup.sh
```

3. Set up the AlphaMapleSAT cubing tool (Use Python 3.10 for optimal compatibility):
```bash
virtualenv --no-download ams_env
source ams_env/bin/activate
cd alphamaplesat
pip install --upgrade pip
pip install -r requirements_cc.txt
```

## Usage

### Entry Point: `parallel-solve.py`

The main script for running the pipeline is `parallel-solve.py`. It supports various configurations for cubing and solving.

#### Example Command:
```bash
python3 parallel-solve.py 17 ks_17.cnf -m 136 --solving-mode satcas --cubing-mode ams --timeout 7200
```

### Metrics Summary: `summary.sh`

After running the pipeline, use `summary.sh` to analyze cubing and solving metrics.

#### Example Command:
```bash
summary.sh <name_of_instance_folder>
```

Ensure each instance has its own folder for better organization.

### Running the AlphaMapleSAT Cubing Tool Independently

To run the cubing tool independently:
```bash
source ams_env/bin/activate
cd alphamaplesat
python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 1 -m 136 -o "test.cubes" -prod
```

This command will generate cubes from the specified constraints file, using a depth of 1 and a maximum of 136 variables and outputting to `test.cubes`. 

## License

This project is licensed under MIT license. See the LICENSE file for details.

## Citation

If you use AlphaMapleSAT in your research, please cite it as follows:

```bibtex
@article{jha2024alphamaplesat,
  title={Alphamaplesat: An MCTS-based cube-and-conquer SAT solver for hard combinatorial problems},
  author={Jha, Piyush and Li, Zhengyu and Lu, Zhengyang and Bright, Curtis and Ganesh, Vijay},
  journal={arXiv preprint arXiv:2401.13770},
  year={2024}
}
```