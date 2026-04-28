import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def _build_train_command(args):
    cmd = [
        args.python,
        "train_brax.py",
        f"+experiment={args.experiment}",
        f"agent.data_collection.address={args.collector_address}",
    ]
    if args.quick:
        cmd.extend(
            [
                "training.wandb_id=null",
                "writers=[stderr]",
                "training.render=false",
                "training.num_timesteps=64",
                "training.num_evals=1",
                "training.num_eval_envs=1",
                "training.num_eval_episodes=1",
                "training.episode_length=32",
                "agent.min_replay_size=32",
                "agent.batch_size=32",
                "agent.model_grad_updates_per_step=1",
                "agent.critic_grad_updates_per_step=1",
                "agent.num_model_rollouts=8",
            ]
        )
    cmd.extend(args.override)
    return cmd


def _terminate_process(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Start the local dummy ZMQ server and launch train_brax with a "
            "localhost hardware collector."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for both server and training commands.",
    )
    parser.add_argument(
        "--experiment",
        default="rccar_sbsrl_hardware_local",
        help="Hydra experiment name passed as +experiment=<name>.",
    )
    parser.add_argument(
        "--collector-address",
        default="tcp://localhost:5555",
        help="Address for train_brax hardware collector override.",
    )
    parser.add_argument(
        "--server-address",
        default="tcp://*:5555",
        help="Bind address for dummy server.",
    )
    parser.add_argument(
        "--server-mode",
        choices=["rccar", "generic"],
        default="rccar",
        help="Dummy server payload mode.",
    )
    parser.add_argument(
        "--server-delay",
        type=float,
        default=0.1,
        help="Artificial response delay in seconds.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use short smoke-test overrides.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=1.0,
        help="Seconds to wait after starting server before launching training.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides. Repeat for multiple values.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    server_script = repo_root / "utility_scripts" / "dummy_zmq_server.py"
    server_cmd = [
        args.python,
        str(server_script),
        "--mode",
        args.server_mode,
        "--address",
        args.server_address,
        "--delay",
        str(args.server_delay),
    ]
    train_cmd = _build_train_command(args)

    print("[local-runner] Starting dummy server:")
    print(" ".join(server_cmd))
    server_proc = subprocess.Popen(server_cmd, cwd=repo_root)

    def _handle_signal(signum, _frame):
        print(f"[local-runner] Received signal {signum}, stopping server...")
        _terminate_process(server_proc)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        time.sleep(args.startup_wait)
        if server_proc.poll() is not None:
            raise RuntimeError("Dummy server exited before training started.")
        print("[local-runner] Launching training:")
        print(" ".join(train_cmd))
        result = subprocess.run(train_cmd, cwd=repo_root)
        raise SystemExit(result.returncode)
    finally:
        print("[local-runner] Stopping dummy server...")
        _terminate_process(server_proc)


if __name__ == "__main__":
    main()
