
# Safe Learning on Real Robots
A collection of algorithms and experiment tools for safe sim to real transfer and learning in robotics.

## Features
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

1. From the parent directory of `safe-sim2real`, clone the repository and check
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

## Usage

Our code uses [Hydra](https://hydra.cc/) to configure experiments. Each experiment is defined as a `yaml` file in `ss2r/configs/experiments`. For example, to train a Unitree Go1 policy with a constraint on joint limit:

```bash
python train_brax.py +experiment=go1_sim_to_real
```
## Docs
* Policies (in `onnx` format) used for the Unitree Go1 experiments can be found in `ss2r/docs/policies`.
* In `ss2r/docs/videos` you can find videos of 5 trials for each policy, marked by its policy id.


## Anonymous Submission

Citation details are omitted in this anonymous submission version.

<!-- ## Learn More

* **Project Webpage**: 
* **Paper**:
* **Contact**: 
