# MeATCube - EnergyCompress

Coming soon:
- Anaconda package
- Pip package

The `inamoriISel` (for "**inamori**1932's **i**nstance **sel**ection") folder is taken from https://github.com/inamori1932/instance-selection-approaches ([permalink](https://github.com/inamori1932/instance-selection-approaches/tree/a857d9c88a9d92e55083f8607932502cb989d7bf/src/main/python/iSel)), that is under MIT License. Minor changes and a wrapper was produced to make the models compatible with the EnergyCompress experimental framework.
Many thanks to the authors for making their code accessible.

How to cite:
```bib
@inproceedings{energycompress,
    note={To be added upon acceptance}
}
```

Content:
- [Download](#download)
  - [In a Git repository: add as a submodule](#in-a-git-repository-add-as-a-submodule)
  - [Clone as a folder](#clone-as-a-folder)
- [Install dependencies](#install-dependencies)
  - [Pip](#pip)
  - [Venv](#venv)
  - [Conda \& CPU](#conda--cpu)
  - [Conda \& GPU (exemple)](#conda--gpu-exemple)
- [Usage](#usage)
  - [Reproduce the experiments of the paper](#reproduce-the-experiments-of-the-paper)
  - [In your own code](#in-your-own-code)


## Download
### In a Git repository: add as a submodule
First time:
```bash
git submodule add https://github.com/EMarquer/MeATCube.git
```

After cloning the parent repository to which MeATCube was added, run `git submodule init`:
```bash
git clone my_fancy_repo.git
git submodule init
```
Or, more simply:
```bash
git clone --recurse-submodules my_fancy_repo.git
```

Then, to update the MeATCube code to the latest version:
```bash
git submodule update --remote MeATCube
```

Have fun!

### Clone as a folder
```bash
git clone https://github.com/EMarquer/MeATCube.git
```

## Install dependencies
The code relies on Python 3.10.
### Pip
`pip install -r requirements.txt`

### Venv
- `pip install virtualenv` (if you don't already have virtualenv installed)
- `virtualenv venv` to create your new environment (called 'venv' here)
- `source venv/bin/activate` to enter the virtual environment
- `pip install -r requirements.txt`

### Conda & CPU
`conda create --name meat python=3.10  --file requirements.txt`

### Conda & GPU (exemple)
Run:
- `conda create --name meat python=3.10 --file requirements-gpu.txt`
- use the suitable install command from [PyTorch - Getting Started](https://pytorch.org/get-started/locally/), for example (Aug. 1st 2023): `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

## Usage
### Reproduce the experiments of the paper
1. Activate the venv or conda environment created in [Install dependencies](#install-dependencies).
2. Prepare the datasets using the preprocessing code in [`benchmark/maintenance/preprocess/`](benchmark/maintenance/preprocess/).
    To do so, run all the preprocessing files using:
    ```bash
    ./benchmark_preprocess_all.sh
    ```
    that will run one by one the scripts in [`benchmark/maintenance/preprocess/`](benchmark/maintenance/preprocess/) that correspond to each dataset.
3. Fit all the models at one using:
    ```bash
    python benchmark/benchmark_all.py
    ```
    or run them one by one using:
    ```bash
    python benchmark/benchmark_one.py
    ```
    *see `python benchmark/benchmark_one.py --help` for information on the parameters and options*
4. Generate the summary HTML table [`benchmark/results/summary.html`](benchmark/results/summary.html) by running:
    ```bash
    python benchmark/summary_table.py
    ```
5. Analyze the results with the jupyter notebooks in [`examples/expe_analysis_ijcai/`](examples/expe_analysis_ijcai/)

### In your own code
Add the following in your python files before trying to import MeATCube:
```python
MEATCUBE_PATH = "MeATCube/meatcube2"
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), MEATCUBE_PATH))
```

Change the value of `rel_path_to_meatcube = ...` so that it contains the relative path to the `meatcube` folder of this repository. The example above is for a `.py` file in the same folder as the folder where MeATCube was cloned.

This process will be simplified in later versions when a package will be made available.