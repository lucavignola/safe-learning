
# SBSRL: Deep experiments

This anonymous repository contains the code for SBSRL and the deep reinforcement learning experiments presented in the paper, along with implementations of other baseline algorithms for comparison. The codebase is designed for both simulation-based experiments and real robot deployment.

The next sections provide instructions for reproducing the deep reinforcement learning experiments presented in the paper.

### Running Experiments

All experiments are configured via [Hydra](https://hydra.cc/) configuration files located in `ss2r/configs/experiment/`. To train a policy with a specific configuration, use the `train_brax.py` script:

```bash
python train_brax.py +experiment=<experiment_name>
```

#### Offline-to-Online experiments in Simulation

For reproducing the offline-to-online experiments described in Appendix of the paper, the following pipeline must be followed
1. For the desired environment, launch the config in `ss2r/configs/experiment/` that contains `collect` (or `simple`) in its name, making sure that the flag `store_buffer` is set to true. This will collect the prior data that will be used to initialize our offline-to-online pipeline.
2. Launch the config in `ss2r/configs/experiment/` that contains `offline` in its name, making sure that the flag `offline` is set to true, and modifying `training.wandb_id` so that it points to the artifact of the run that was used to collect data in the previous step. This will run SBSRL in offline mode, effectively building the prior that will be used to initialize our online algorithm and all the baselines.
3. Finally launch the online version of SBSRL or the desired baseline, using the remaining configs in `ss2r/configs/experiment/` and setting `training.wandb_id` to target the artifact of the offline run. For the baselines, e.g. MBPO and SOOPER, make sure to set the flag `load_from_sbsrl` to true, so as to start from the same offline prior as SBSRL, ensuring fair comparison.
   

#### Hardware Experiments

The code for reproducing the hardware experiments presented in the paper can be found in `ss2r/rccar_experiments/`. Note that `ss2r/train_brax.py` can be launched on a separate machine where training will be performed, using the configs in `ss2r/configs/experiment/` containing the keyword `hardware`. Launching `ss2r/rccar_experiments/online_learning.py` on the real robot will then rely on `ss2r/rl/online.py` to deploy the policy on hardware.
Finally, note that the offline-to-online pipeline described in the previous subsection must be followed also for the hardware experiments.


### Hyperparameter Details

All hyperparameters used in the experiments can be found in the corresponding configuration files under `ss2r/configs/`. The configuration hierarchy includes:
- **Agent configs** (`ss2r/configs/agent/`): Algorithm-specific parameters (SBSRL, CRPO, Saute-RL, SAC, MBPO, PPO)
- **Benchmark configs** (`ss2r/configs/environment/`): Environment-specific settings
- **Experiment configs** (`ss2r/configs/experiment/`): Complete experiment definitions combining agent and benchmark configurations
- **Hydra configs** (`ss2r/configs/hydra/`): System-level settings (launcher, job logging)

Each configuration file is self-documenting and contains all parameters necessary to reproduce the reported results.

### Installation Notes

**Note**: For this paper you can skip the optional `madrona_mjx` installation step described in the [Installation](#installation) section. The basic installation via `pip` or `uv` is sufficient.

### Environment Details

The experiments require Python 3.11.6 and the dependencies listed in `pyproject.toml`. The environment can be set up following the [Installation](#installation) instructions below.

## Additional Features
* Three different CMDP solvers, [CRPO](https://arxiv.org/abs/2011.05869), [Saute-RL](https://arxiv.org/abs/2202.06558) and primal-dual, compatible with (variants of) [Brax's](https://github.com/google/brax) SAC, MBPO and PPO.
* Algorithm implementation is interchangeable between training in simulation to training on real robots via [OnlineEpisodeOrchestrator](ss2r/rl/online.py). Check out `rccar_experiments` for a full example. Support for training online on any real robot supported by [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), including Unitree Go1/2.
* Fast training. Full compatibility with [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground). Reimplementation of OpenAI's [Safety Gym](https://github.com/openai/safety-gym) in MJX and safety tasks from [Real-World RL suite](https://github.com/google-research/realworldrl_suite/tree/master).


## Requirements

- Python == 3.11.6
- [`uv`](https://docs.astral.sh/uv/) (recommended) or the built-in `venv`

## Installation

### Using pip

```bash
git clone <anonymous-repo-url>
cd safe-learning
python3 -m venv venv
source venv/bin/activate
pip install -e .
````

### Using uv

Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a project environment and install dependencies:

```bash
git clone <anonymous-repo-url>
cd safe-learning
uv sync
uv run python --version  # sanity check, optional
```

### Installing `madrona_mjx` (optional, required for Madrona backend)

Some benchmarks (e.g., the MJX-based pick-and-place tasks) require the custom
[`madrona_mjx`](https://github.com/shacklettbp/madrona_mjx) fork. Build and
install it inside the UV environment you created above:

1. From the parent directory of `<anonymous-repo>`, clone the repository and check
   out the tested commit:

   ```bash
   git clone https://github.com/shacklettbp/madrona_mjx.git
   cd madrona_mjx
   git checkout c34f3cf6d95148dba50ffeb981aea033b8a4d225
   git submodule update --init --recursive
   ```

2. Configure and build (disable Vulkan if you do not have it available):

   ```bash
   mkdir -p build
   cd build
   cmake -DLOAD_VULKAN=OFF ..
   cmake --build . -j
   cd ..
   ```

3. While having your environment activated, install the Python bindings into your UV environment:

   ```bash
   uv pip install -e .
   ```

Refer to the upstream repository for platform-specific prerequisites (CUDA,
Vulkan, compiler versions). Re-run `uv pip install -e .` whenever you rebuild the
library.

**Troubleshooting tips**

- If you see CUDA OOMs immediately after the build, try `export
  MADRONA_DISABLE_CUDA_HEAP_SIZE=1` before launching training.
- Populate the kernel caches to avoid recompilation on every run:

  ```bash
  export MADRONA_MWGPU_KERNEL_CACHE=/path/to/cache/mwgpu
  export MADRONA_BVH_KERNEL_CACHE=/path/to/cache/bvh
  ```


##  Citation

Citation details are omitted in this anonymous submission version.

<!-- ## Learn More

* **Project Webpage**: 
* **Paper**:
* **Contact**: 
