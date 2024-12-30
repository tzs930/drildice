# Implementation Code for [DrilDICE (NeurIPS 2024)](https://openreview.net/pdf?id=lHcvjsQFQq)

- This code contains official implementation codes of DrilDICE  and baselines for offline imitation learning.

### 1. Prerequisites

- To run this code, first install the anaconda virtual environment and install D4RL:

```
conda env create -f environment.yml
conda activate drildice
pip install d4rl
```

- (optional) Download D4RL dataset:
```
python download_d4rl_dataset.py
```

### 2. Generate Subsampling Indices for Covariate Shift Scenarios
```
python generate_subsmaple_indices.py
```

### 3. Train & Evaluate DrilDICE
- Train imitation policies using `main.py`.

### Toy Domain:
- To run Four Rooms env., you should get MOSEK license.
```
pip install cvxpy[mosek] mosek
```
- Run `cd fourrooms; python run_toy.py`
