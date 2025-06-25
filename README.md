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
