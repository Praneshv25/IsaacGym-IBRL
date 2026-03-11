## ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning
---

### Install Conda

First, install Conda by following the instructions on the [Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) (here using Miniconda).

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

To activate the changes, restart your shell or run:

```bash
source ~/.bashrc
source ~/.zshrc
```
### Create a Conda Environment
- Create a python 3.8 environment e.g. with a conda virtual env.
```commandline
conda create --name tacsl python==3.8 -y
```

- Download TacSL-specific Isaac Gym binary from the shared drive [here](https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX)
and pip install within python 3.8:
```commandline
pip install -e IsaacGym_Preview_TacSL_Package/isaacgym/python/
```

- Install the `manifeel-tacsl` branch from [here](https://github.com/quan-luu/manifeel-isaacgymenvs/tree/tacsl-manifeel-rl):
```commandline
pip install -e isaacgymenvs
```

- Install additional dependencies from the [requirements.txt](https://github.com/quan-luu/manifeel-isaacgymenvs/blob/manifeel-tacff/isaacgymenvs/tacsl_sensors/install/requirements.txt):
```commandline
pip install -r requirements.txt
```

---
## Running IsaacGym Tasks

- To confirm that installing IsaacGym is successful, try running the simple examples e.g.:
```commandline
cd IsaacGym_Preview_TacSL_Package/isaacgym/python/examples
python franka_osc.py
```

---
## Running task specific to Manifeel TaskSuite

- To confirm that installing Manifeel-IsaacEnv is successful, try running the simple examples e.g.:
- Download the `demo.py` script from [here](https://github.com/quan-luu/manifeel-isaacgymenvs/blob/manifeel-tacff/examples/demo.py):
```commandline
cd manifeel-isaacgymenvs/examples
python demo.py
```

---

## Frequently Asked Questions

1) If you encounter error running IsaacGym: `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`

```commandline
export LD_LIBRARY_PATH=/home/${USER}/anaconda3/envs/tacsl/lib:$LD_LIBRARY_PATH
```

2) For python-related error, confirm that the packages in the [requirements.txt](https://github.com/quan-luu/manifeel-isaacgymenvs/blob/manifeel-tacff/isaacgymenvs/tacsl_sensors/install/requirements.txt) are installed with the specified versions.
