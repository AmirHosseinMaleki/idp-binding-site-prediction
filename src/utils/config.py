"""
src/utils/config.py
-------------------
Loads config.yaml and resolves embedding / model paths so every script
can do:

    from src.utils.config import load_config, get_embedding_path, get_model_path

    cfg = load_config()                            # uses default config.yaml
    cfg = load_config("path/to/other.yaml")        # explicit override

    emb = get_embedding_path(cfg, "ahojdb_train")  # full absolute path
    mdl = get_model_path(cfg, "protein_hybrid")    # full absolute path
"""

import os
import yaml
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Walk upward from this file until we find config.yaml.
    Falls back to the current working directory.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


def load_config(config_path: str | None = None) -> dict:
    """
    Load and return the YAML configuration dictionary.

    Parameters
    ----------
    config_path : str or None
        Explicit path to a config YAML file.  When None, the function looks
        for config.yaml in the project root (detected automatically).

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If no config file is found.
    """
    if config_path is None:
        root = _find_project_root()
        config_path = root / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Either pass --config <path> on the command line or place "
            "config.yaml in the project root."
        )

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    return cfg


# ---------------------------------------------------------------------------
# Path resolvers
# ---------------------------------------------------------------------------

def get_embedding_path(cfg: dict, key: str) -> str:
    """
    Return the absolute path for an embedding file.

    The key must match one of the entries under cfg["embeddings"] (e.g.
    "ahojdb_train", "scannet_val", "disprot_ion_test").

    The filename is joined with cfg["embeddings"]["dir"] so that only the
    directory needs to be changed when moving data.
    """
    emb_cfg = cfg["embeddings"]
    filename = emb_cfg[key]          # e.g. "ahojdb_train_embeddings.npz"
    return str(Path(emb_cfg["dir"]) / filename)


def get_model_path(cfg: dict, key: str) -> str:
    """
    Return the absolute path for a saved model weight file.

    The key must match one of the entries under cfg["models"] (e.g.
    "protein_hybrid", "ion_phase1").
    """
    mdl_cfg = cfg["models"]
    filename = mdl_cfg[key]          # e.g. "protein_hybrid_idpval_model.pt"
    return str(Path(mdl_cfg["dir"]) / filename)


def get_dataset_path(cfg: dict, *keys) -> str:
    """
    Return the absolute path for a dataset file by traversing nested keys.

    Examples
    --------
    get_dataset_path(cfg, "scannet", "train_clustered_csv")
    get_dataset_path(cfg, "disprot", "protein_train_tsv")
    get_dataset_path(cfg, "biolip", "dna_train_csv")
    """
    node = cfg["datasets"]
    for k in keys:
        node = node[k]
    return str(node)


# ---------------------------------------------------------------------------
# Argparse helpers  (for use in every script's argument parser)
# ---------------------------------------------------------------------------

def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add a --config argument to an existing ArgumentParser.
    Call this before parser.parse_args().
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to config.yaml.  Defaults to config.yaml in the project "
            "root (auto-detected).  Use this to run with a different "
            "environment or dataset layout."
        ),
    )
    return parser


def base_parser(description: str = "") -> argparse.ArgumentParser:
    """
    Return a base ArgumentParser that already includes --config.
    Scripts can add their own arguments on top.

    Example
    -------
    parser = base_parser("Train ion binding model - phase 1")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg  = load_config(args.config)
    """
    parser = argparse.ArgumentParser(description=description)
    add_config_argument(parser)
    return parser