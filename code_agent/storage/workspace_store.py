"""
Resolves where repos are cloned based on deployment mode.

local: /tmp/code_agent_workspaces  (ephemeral, per-pod)
eks:   /mnt/efs/code_agent_workspaces  (EFS PVC, shared across pods)
       or WORKSPACE_DIR env override
"""
import os
import logging

logger = logging.getLogger(__name__)

def get_workspace_base_dir() -> str:
    """
    Returns the base directory for repo clones.

    In EKS, this should be an EFS-backed PVC mount so that:
    - Multiple pods share the same cloned repos
    - Clones survive pod restarts
    - A repo only needs to be cloned once per workspace crawl

    Set WORKSPACE_DIR explicitly to override.
    """
    explicit = os.getenv("WORKSPACE_DIR", "").strip()
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit

    mode = os.getenv("DEPLOYMENT_MODE", "local").lower()
    if mode == "eks":
        efs_path = "/mnt/efs/code_agent_workspaces"
        try:
            os.makedirs(efs_path, exist_ok=True)
            logger.info("Workspace: using EFS path %s", efs_path)
            return efs_path
        except OSError:
            logger.warning("EFS path %s not writable, falling back to /tmp", efs_path)

    tmp_path = "/tmp/code_agent_workspaces"
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path
