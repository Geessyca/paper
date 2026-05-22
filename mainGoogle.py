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


def write_drive_probe(backup_dir: str) -> None:
    test_path = os.path.join(backup_dir, "drive_backup_test.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("drive backup ok\n")


def run_colab() -> None:
    drive_root = mount_drive()
    config = load_config("config/default.yaml")
    if drive_root:
        backup_dir = os.path.join(drive_root, "paper", config["logging"]["output_dir"])
        config["logging"]["backup_dir"] = backup_dir
        ensure_dir(backup_dir)
        write_drive_probe(backup_dir)

    main(config)


if __name__ == "__main__":
    run_colab()