import os

from config import load_config
from dqn_env import ensure_dir
from main import main


def mount_drive() -> str:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        return "/content/drive/My Drive"
    except Exception:
        return ""


def run_colab() -> None:
    drive_root = mount_drive()
    config = load_config("config/default.yaml")
    if drive_root:
        output_dir = os.path.join(drive_root, "paper", config["logging"]["output_dir"])
        config["logging"]["output_dir"] = output_dir
        ensure_dir(output_dir)

    main(config)


if __name__ == "__main__":
    run_colab()