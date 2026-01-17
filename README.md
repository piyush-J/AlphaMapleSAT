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

## Benchmarks Used in the Paper

The following benchmarks were used in the experimental evaluation. Each benchmark includes a link to the corresponding repository or documentation and the command used to generate problem instances.

### Kochen–Specker (SAT+CAS)

* **Repository:** [MathCheck](https://github.com/BrianLi009/MathCheck)
* **Instance generation:**

```bash
python gen_instance/generate.py n 0.5
````

### Ramsey R(8,3) (SAT+CAS)

* **Repository:** [MathCheckRamsey](https://github.com/ConDug/MathCheckRamsey)
* **Instance generation:**

```bash
./main.sh -n --deg-card totalizer --strict-degree-bound 28 8 3
```

### Kochen–Specker (SMS)

* **Repository:** [sat-modulo-symmetries](https://github.com/markirch/sat-modulo-symmetries)
* **Instance generation:**

```bash
python ./encodings/kochen_specker.py -v n
```

### Diameter-2 Critical Graphs (SMS)

* **Documentation:** [PySMS GraphEncodingBuilder](https://sat-modulo-symmetries.readthedocs.io/en/latest/reference/#pysms.graph_builder.GraphEncodingBuilder.diameter2critical)
* **Instance generation:**

```bash
python3 -m pysms.graph_builder --vertices 10 --diam2-critical --partial-sym-break
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


