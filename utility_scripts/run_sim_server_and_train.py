import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


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


def _build_server_command(python_exec, online_config):
    return [
        python_exec,
        "rccar_experiments/online_learning.py",
        "--config-name",
        online_config,
        "+mode=sim",
    ]


def _build_train_command(python_exec, train_experiment):
    return [
        python_exec,
        "train_brax.py",
        f"+experiment={train_experiment}",
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Start online_learning with a config, then launch train_brax with an "
            "experiment config."
        )
    )
    parser.add_argument(
        "--online-config",
        default="rccar_online_learning_sbsrl",
        help="Hydra config name for rccar_experiments/online_learning.py.",
    )
    parser.add_argument(
        "--train-experiment",
        default="rccar_sbsrl_hardware_local",
        help="Hydra experiment for train_brax.",
    )
    args = parser.parse_args()

    python_exec = sys.executable
    startup_wait = 1.5
    repo_root = Path(__file__).resolve().parents[1]
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    root_str = str(repo_root)
    child_env["PYTHONPATH"] = (
        f"{root_str}:{existing_pythonpath}" if existing_pythonpath else root_str
    )
    server_cmd = _build_server_command(python_exec, args.online_config)
    train_cmd = _build_train_command(python_exec, args.train_experiment)

    print("[sim-runner] Starting sim server:")
    print(" ".join(server_cmd))
    server_proc = subprocess.Popen(server_cmd, cwd=repo_root, env=child_env)

    def _handle_signal(signum, _frame):
        print(f"[sim-runner] Received signal {signum}, stopping server...")
        _terminate_process(server_proc)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        time.sleep(startup_wait)
        if server_proc.poll() is not None:
            raise RuntimeError("Sim server exited before training started.")
        print("[sim-runner] Launching train_brax:")
        print(" ".join(train_cmd))
        train_proc = subprocess.Popen(train_cmd, cwd=repo_root, env=child_env)
        while True:
            train_rc = train_proc.poll()
            server_rc = server_proc.poll()
            if train_rc is not None:
                raise SystemExit(train_rc)
            if server_rc is not None:
                _terminate_process(train_proc)
                raise RuntimeError(
                    f"Sim server exited during training with code {server_rc}."
                )
            time.sleep(1.0)
    finally:
        print("[sim-runner] Stopping sim server...")
        _terminate_process(server_proc)


if __name__ == "__main__":
    main()
