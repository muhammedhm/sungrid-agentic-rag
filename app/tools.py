"""
Tools available to the agent.

`check_eligibility` is a *thin* wrapper around
`app.stub_tools.check_rebate_eligibility` -- the interface and eligibility
logic live entirely in the stub, per the assessment brief. This wrapper
only adds a LangChain-compatible schema so the LLM can call it directly.
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.stub_tools import EligibilityResult, check_rebate_eligibility


def format_docs_for_context(docs: List[Document]) -> str:
    """Render retrieved chunks with citation markers the LLM (and we) can trace back to a source file."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(blocks)


class CheckEligibilityArgs(BaseModel):
    household_zip: str = Field(..., description="5-digit household ZIP code, e.g. '94101'.")
    annual_income_usd: float = Field(..., description="Household adjusted gross annual income in USD.")
    system_size_kw: float = Field(..., description="Proposed or installed rooftop system size in kW.")
    installer_approved: bool = Field(
        ..., description="Whether the installer performing the work is on SunGrid's approved installer list."
    )


@tool("check_rebate_eligibility", args_schema=CheckEligibilityArgs)
def check_eligibility_tool(
    household_zip: str,
    annual_income_usd: float,
    system_size_kw: float,
    installer_approved: bool,
) -> EligibilityResult:
    """
    Run SunGrid's Rooftop Rebate Program hard-eligibility checks (ZIP,
    income, system size, installer approval) and return whether the
    household is eligible plus an estimated rebate amount.

    Only call this once you have all four inputs. If any are missing or
    unclear, ask the user for them instead of guessing or assuming a
    default value.
    """
    return check_rebate_eligibility(
        household_zip=household_zip,
        annual_income_usd=annual_income_usd,
        system_size_kw=system_size_kw,
        installer_approved=installer_approved,
    )


ALL_TOOLS = [check_eligibility_tool]
