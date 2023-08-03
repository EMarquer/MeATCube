# MeATCube

Coming soon:
- Anaconda package
- Pip package

## Install
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

## Install Dependencies

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
Add the following in your python files before trying to import MeATCube:
```python
rel_path_to_meatcube = "MeATCube/meatcube"
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), rel_path_to_meatcube))
```

Change the value of `rel_path_to_meatcube = ...` so that it contains the relative path to the `meatcube` folder of this repository. The example above is for a `.py` file in the same folder as the folder where MeATCube was clonned.

This process will be simplified in later versions when a package will be made available.