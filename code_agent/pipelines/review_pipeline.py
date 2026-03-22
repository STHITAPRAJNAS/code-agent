"""
PR Review Pipeline — SequentialAgent that chains VCS fetch → navigation → review → security.
Use this for thorough automated PR reviews.
"""
from google.adk.agents import SequentialAgent

from code_agent.agents.vcs_agent import make_vcs_agent
from code_agent.agents.code_navigator import make_code_navigator_agent
from code_agent.agents.pr_reviewer import make_pr_reviewer_agent
from code_agent.agents.security_agent import make_security_agent

# Each agent saves its output to state so the next can read it.
# The orchestrator injects: state["pr_url"] or state["pr_id"] before invoking.

pr_review_pipeline = SequentialAgent(
    name="pr_review_pipeline",
    description="End-to-end PR review: fetch diff, understand context, review, security scan",
    sub_agents=[
        make_vcs_agent(),           # output_key="pr_context"
        make_code_navigator_agent(), # reads pr context, output_key="code_context"
        make_pr_reviewer_agent(),   # reads both, output_key="review_result"
        make_security_agent(),      # reads diff, output_key="security_findings"
    ],
)
