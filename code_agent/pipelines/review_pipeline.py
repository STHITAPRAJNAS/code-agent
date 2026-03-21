"""
PR Review Pipeline — SequentialAgent that chains VCS fetch → navigation → review → security.
Use this for thorough automated PR reviews.
"""
from google.adk.agents import SequentialAgent

from code_agent.agents.vcs_agent import vcs_agent
from code_agent.agents.code_navigator import code_navigator_agent
from code_agent.agents.pr_reviewer import pr_reviewer_agent
from code_agent.agents.security_agent import security_agent

# Each agent saves its output to state so the next can read it.
# The orchestrator injects: state["pr_url"] or state["pr_id"] before invoking.

pr_review_pipeline = SequentialAgent(
    name="pr_review_pipeline",
    description="End-to-end PR review: fetch diff, understand context, review, security scan",
    sub_agents=[
        vcs_agent,              # output_key="pr_context"
        code_navigator_agent,   # reads pr context, output_key="code_context"
        pr_reviewer_agent,      # reads both, output_key="review_result"
        security_agent,         # reads diff, output_key="security_findings"
    ],
)
