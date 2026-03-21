"""
Feature Implementation Pipeline:
1. Understand the affected codebase
2. Implement the feature
3. Update docs/tests
4. Commit changes
5. Create PR
"""
from google.adk.agents import SequentialAgent

from code_agent.agents.vcs_agent import vcs_agent
from code_agent.agents.code_navigator import code_navigator_agent
from code_agent.agents.code_writer import code_writer_agent
from code_agent.agents.docs_agent import docs_agent
from code_agent.agents.git_agent import git_agent

feature_pipeline = SequentialAgent(
    name="feature_pipeline",
    description="End-to-end feature implementation: Jira → understand → implement → test → PR",
    sub_agents=[
        code_navigator_agent,   # understand codebase
        code_writer_agent,      # implement feature
        docs_agent,             # update docs/tests
        git_agent,              # commit changes
        vcs_agent,              # create PR
    ],
)
