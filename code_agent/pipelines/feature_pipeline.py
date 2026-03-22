"""
Feature Implementation Pipeline:
1. Understand the affected codebase
2. Implement the feature
3. Update docs/tests
4. Commit changes
5. Create PR
"""
from google.adk.agents import SequentialAgent

from code_agent.agents.code_navigator import make_code_navigator_agent
from code_agent.agents.code_writer import make_code_writer_agent
from code_agent.agents.docs_agent import make_docs_agent
from code_agent.agents.git_agent import make_git_agent
from code_agent.agents.vcs_agent import make_vcs_agent

feature_pipeline = SequentialAgent(
    name="feature_pipeline",
    description="End-to-end feature implementation: Jira → understand → implement → test → PR",
    sub_agents=[
        make_code_navigator_agent(), # understand codebase
        make_code_writer_agent(),    # implement feature
        make_docs_agent(),           # update docs/tests
        make_git_agent(),            # commit changes
        make_vcs_agent(),            # create PR
    ],
)
