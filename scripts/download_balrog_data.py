"""Download data files needed by Balrog environments.

- Boxoban levels for MiniHack
- TextWorld game files
"""

import os
import subprocess
import sys
import zipfile


def download_boxoban_levels():
    """Download Boxoban levels for MiniHack."""
    try:
        import pkg_resources
        dest = pkg_resources.resource_filename("minihack", "dat")
    except Exception:
        print("minihack not installed, skipping Boxoban levels")
        return

    if os.path.exists(os.path.join(dest, "boxoban-levels-master")):
        print("Boxoban levels already exist, skipping")
        return

    print("Downloading Boxoban levels...")
    url = "https://github.com/deepmind/boxoban-levels/archive/refs/heads/master.zip"
    zip_path = os.path.join(dest, "master.zip")
    os.makedirs(dest, exist_ok=True)
    ret = os.system(f'wget -c --read-timeout=5 --tries=3 "{url}" -O "{zip_path}"')
    if ret != 0:
        print("WARNING: wget failed, trying with curl...")
        ret = os.system(f'curl -L --retry 3 -o "{zip_path}" "{url}"')
    if ret == 0 and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        os.remove(zip_path)
        print("Boxoban levels downloaded and extracted")
    else:
        print("WARNING: Failed to download Boxoban levels")


def _default_balrog_dir():
    """Locate the benchmark.balrog package directory (where tw_games is read from)."""
    try:
        import importlib.resources as ilr
        return str(ilr.files("benchmark.balrog"))
    except Exception:
        # Fallback for when src/ is on sys.path but benchmark is not installed
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "src", "benchmark", "balrog"))


def download_textworld_games(output_dir=None):
    """Download pre-generated TextWorld game files.

    Files are read by `benchmark.balrog.environments.textworld.base.TextWorldFactory`
    via `importlib.resources.files("benchmark.balrog") / "tw_games"`, so the default
    target is the package's own directory.
    """
    if output_dir is None:
        output_dir = _default_balrog_dir()
    target = os.path.join(output_dir, "tw_games")
    if os.path.exists(target) and os.listdir(target):
        print(f"TextWorld games already exist at {target}, skipping")
        return

    try:
        import requests
    except ImportError:
        print("requests not installed, skipping TextWorld games")
        return

    print("Downloading TextWorld games...")
    url = "https://drive.google.com/uc?export=download&id=1aeT-45-OBxiHzD9Xn99E5OvC86XmqhzA"
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except Exception as e:
        print(f"WARNING: Failed to download TextWorld games: {e}")
        return

    zip_path = os.path.join(os.path.abspath(output_dir), "tw-games.zip")
    os.makedirs(output_dir, exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(response.content)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
    except Exception as e:
        print(f"WARNING: Failed to extract TextWorld zip: {e}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    print(f"TextWorld games downloaded to {target}")


def main():
    download_boxoban_levels()
    download_textworld_games()


if __name__ == "__main__":
    main()
