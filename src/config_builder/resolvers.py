"""Custom OmegaConf Resolvers"""
import hashlib

from omegaconf import OmegaConf

from src.utils import utils


def generate_id(description: str, git_issue_id: str, seed: int) -> str:
    return f"{hashlib.sha224(description.encode()).hexdigest()}_issue_{git_issue_id}_seed_{seed}"


def register_new_resolvers():
    OmegaConf.register_new_resolver("git_commit_id", utils.get_current_commit_id)
    OmegaConf.register_new_resolver(
        "git_has_uncommitted_changes", utils.has_uncommitted_changes
    )
    OmegaConf.register_new_resolver("date", utils.get_current_time)
    OmegaConf.register_new_resolver("slurm_id", utils.get_slurm_id)
    OmegaConf.register_new_resolver("generate_id", generate_id)
