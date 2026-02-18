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
`-debug` to see detailed timings, per-simulation MCTS trace (UCT details + rewards + chosen variable), and rankings (rankings are printed only for free/preselected variables).

Note: in the simplified C++ demo, `-d` limits only output cube depth; MCTS itself is not depth-capped by `-d`:

```bash
./simple_mcts alphamaplesat/constraints_17_c_100000_2_2_0_final.simp -m 20 -o out.cubes -numMCTSSims 2 -debug
```

Typical debug output looks like:

```
20 variables will be considered for cubing
No. of free variables: 3
Free variables: 1 3 5
Variable ranking with normalization (var:raw:norm_raw:pos:neg:norm_pos:norm_neg):
1. 5:323.000:1.000000:12:10:1.000000:0.833333
2. 3:200.000:0.619195:7:8:0.583333:0.666667
=== MCTS simulation 1/2 ===
[sim 1][depth 0] top actions by UCT (max 3 candidates):
Parsing time: 0.001
Scoring time: 0.002
MCTS time: 0.001
Cube gen time: 0.000
Write time: 0.000
Accounted component time: 0.004
Time taken for cubing:  0.003
Number of nodes:  23
Tool runtime:  0.681
```


### Beam lookahead demo

A standalone beam-lookahead prototype is provided in `beam_lookahead.cpp`.
It performs:
1. Score all variables on `F` with `score = pos*neg + pos + neg`.
2. Keep top-3 seed variables.
3. Build six sub-formulas via unit assumptions: `F∧x1`, `F∧-x1`, `F∧x2`, `F∧-x2`, `F∧x3`, `F∧-x3`.
4. For each seed `x`, compute depth-1 and depth-2 signals:
   - `S1(x)=score(x|base=[])`.
   - `S2+(x)=score(x_j+|base=[x])`, `S2-(x)=score(x_j-|base=[-x])` where each `x_j` is the best partner in that signed sub-formula.
   - `S2_balanced(x)=mean(S2+,S2-) - gamma*|S2+-S2-|` (risk-aware).
5. Normalize per depth across the top-3 seeds and combine:
   `updated(x)=(1-lambda)*S1_norm + lambda*S2_norm`.
6. Pick best seed and write a 1-variable cube file:
   `a -x 0` and `a x 0`.

Build and run:

```bash
g++ -std=c++17 -O2 beam_lookahead.cpp -o beam_lookahead
./beam_lookahead alphamaplesat/constraints_17_c_100000_2_2_0_final.simp -m 20 -o out.cubes -debug
```

Without `-debug`, beam lookahead prints a concise summary plus timing stats:

```
Beam-lookahead scoring on first 136 variables
Top-3 variables on original formula:
1) var=79 pos=26 neg=19 score=539.000
2) var=80 pos=19 neg=26 score=539.000
3) var=67 pos=71 neg=6 score=503.000

Best variable after beam lookahead: x=67 updated_score=0.966605
Saved cubes to file  out.cubes
Parse time: 0.522
Initial scoring time: 1.973
Lookahead time: 15.030
Write time: 0.000
Total runtime: 17.525
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
