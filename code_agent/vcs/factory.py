"""
factory.py — Factory function for constructing VCS provider instances.

Imports are deferred to avoid pulling in optional dependencies (PyGithub,
requests, etc.) until the specific provider is actually needed.
"""

from __future__ import annotations

from code_agent.vcs.base import VCSProvider


def create_vcs_provider(vcs_type: str, **kwargs) -> VCSProvider:
    """
    Instantiate and return the appropriate :class:`VCSProvider` subclass.

    Args:
        vcs_type: One of ``"github"``, ``"bitbucket-cloud"``, or
                  ``"bitbucket-server"``.
        **kwargs: Credentials and configuration forwarded to the provider's
                  ``__init__``.

                  * **github** — ``token: str``, ``org: str | None``
                  * **bitbucket-cloud** — ``username: str``,
                    ``app_password: str``, ``workspace: str``
                  * **bitbucket-server** — ``base_url: str``, ``token: str``,
                    ``project_key: str``

    Returns:
        A fully initialised :class:`VCSProvider` instance.

    Raises:
        ValueError: When *vcs_type* is not recognised.

    Example::

        from code_agent.vcs import create_vcs_provider

        gh = create_vcs_provider("github", token="ghp_xxx", org="my-org")
        repos = gh.list_repos()
    """
    if vcs_type == "github":
        from code_agent.vcs.github_provider import GitHubProvider

        return GitHubProvider(**kwargs)
    elif vcs_type == "bitbucket-cloud":
        from code_agent.vcs.bitbucket_cloud_provider import BitbucketCloudProvider

        return BitbucketCloudProvider(**kwargs)
    elif vcs_type == "bitbucket-server":
        from code_agent.vcs.bitbucket_server_provider import BitbucketServerProvider

        return BitbucketServerProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown VCS type: {vcs_type!r}. "
            "Must be one of: github, bitbucket-cloud, bitbucket-server"
        )
