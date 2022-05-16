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