# AlphaMapleSAT

AlphaMapleSAT is a novel Monte Carlo Tree Search (MCTS) based Cube-and-Conquer (CnC) SAT solving method aimed at efficiently solving challenging combinatorial problems. 

## Installation

Set up the AlphaMapleSAT cubing tool (Use Python 3.10 for optimal compatibility):

```bash
virtualenv --no-download ams_env
source ams_env/bin/activate
cd alphamaplesat
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

To run the AlphaMapleSAT cubing tool:
```bash
source ams_env/bin/activate
cd alphamaplesat
python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 1 -m 136 -o "test.cubes" -prod
```

This command will generate cubes from the specified constraints file (provided as an example in the repo), using a depth of 1 and a maximum of 136 variables and outputting to `test.cubes`.

## C++ cubing demo

A lightweight C++ version of the cubing procedure is provided in `simple_mcts.cpp`.
The program implements a tiny MCTS engine that selects variable assignments to
form a cube. Compile it with a C++17 compiler:

```bash
g++ -std=c++17 -O2 simple_mcts.cpp -o simple_mcts
```

Run the tool on the sample CNF file with a handful of search simulations. Pass
`-debug` to see detailed timings and rankings:

```bash
./simple_mcts alphamaplesat/constraints_17_c_100000_2_2_0_final.simp -m 20 -o out.cubes -numMCTSSims 2 -debug
```

Typical debug output looks like:

```
20 variables will be considered for cubing
No. of free variables: 3
Free variables: 1 3 5
Variable ranking (var:score):
1. 5:323
2. 3:200
Parsing time: 0.001
Scoring time: 0.002
MCTS time: 0.001
Cube gen time: 0.000
Write time: 0.000
Number of nodes:  23
Tool runtime:  0.681
```

Without `-debug`, only a concise summary is printed:

```
20 variables will be considered for cubing
No. of free variables: 3
Saved cubes to file  out.cubes
Time taken for cubing:  0.000
Number of nodes:  23
Tool runtime:  0.681
```

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
