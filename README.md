# PMLR-Waymo

## Usage instructions

### Activating virtual environment

- `module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy`
- `source $HOME/.local/bin/virtualenvwrapper.sh`
- `workon <virtual environment>`

### Copying data to other user

- `scp -r <folder> <user>@euler.ethz.ch:../../scratch/<user>`
- input user's password

### Working with wandb

- `pip install wandb`
- `TMPDIR=~/logs`
- `export TMPDIR`
- `wandb login`
- input login token from personal wandb account
- move to root directory of project (PMLR-Waymo)
- `wandb init`
- run code
- `cd ~/logs`
- `wandb sync --clean --sync-all`
- runs can be viewed at https://wandb.ai/lrabuzin/pmlr-waymo

### Submitting jobs

- see Euler's documentation
- when calling `python training.py`, the root directory with individual frames needs to be passed as `--root_dir` and directory where model checkpoints will be saved needs to be passed as `--checkpoint_location`
- if you want hyperparameters other than defaults, pass them the same way, using keywords `--batch_size`, `--max_epochs`, `--lr`, `--momentum`, `--weight_decay`