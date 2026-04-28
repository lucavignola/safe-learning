## Using this repository on a cluster
1. Use the provided `setup.bash` script via `source setup.bash`
2. Install the environment with `poetry install`


## Installing Madrona-MJX on a cluster
1. Make sure that the steps above have been completed in the current terminal.
2. Go to the parent directory of the repository: `cd ..`
3. Activate the environment: `source .venv/bin/activate`
4. Move out of the repository to your home folder.
5. `git clone https://github.com/shacklettbp/madrona_mjx.git && git checkout c34f3cf6d95148dba50ffeb981aea033b8a4d225`
6. `cd madrona_mjx`
7. `git submodule update --init --recursive`
8. Load cmake `module load cmake/3.27.7`
9. Point CMAKE_PREFIX_PATH to CUDA without loading CUDA `export CMAKE_PREFIX_PATH="/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/:${CMAKE_PREFIX_PATH}"`
10. `mkdir build && cd build && cmake -DLOAD_VULKAN=OFF .. && cd ..`
11. Compile on a compute node `srun --ntasks=1 --cpus-per-task=20 --gpus=rtx_4090:1 --time=0:30:00 --mem-per-cpu=10240 make -j`
12. Change back to the root of the repository: `cd .. && cd safe-learning`
13. `poetry install --with madrona-mjx`
14. Test `srun --ntasks=1 --cpus-per-task=20 --gpus=rtx_4090:1 --time=0:30:00 --mem-per-cpu=10240 python train_brax.py +experiment=cartpole_vision`