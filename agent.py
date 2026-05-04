"""
Job Monitor Agent
=================
This is a ReAct-style AI agent that:
1. Uses Claude + web_search to find agentic AI job postings
2. Filters/ranks them with an LLM pass
3. Prints a digest to the terminal

HOW THE AGENT LOOP WORKS (this is what happened when Claude Code applied to Medfinder):

  while True:
    response = claude(messages, tools)          # Claude thinks + decides what tool to call
    if response.stop_reason == "end_turn":      # Claude is done
        break
    tool_result = run_tool(response.tool_use)   # We execute the tool
    messages.append(tool_result)                # Claude sees the result
    # loop continues — Claude reacts to the result and decides next action

That's the entire agent loop. Everything else is scaffolding around it.
"""

import anthropic
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# Job search configuration
JOB_SEARCHES = [
    "agentic AI engineer new grad entry level jobs 2026",
    "AI voice agent engineer junior early career jobs 2026",
    "full stack engineer React Node.js Python new grad jobs 2026",
    "backend engineer FastAPI Python entry level jobs 2026",
]

# ============================================================
# STEP 1: THE ANTHROPIC CLIENT
# ============================================================
client = anthropic.Anthropic()


# ============================================================
# STEP 2: TOOL DEFINITIONS
# ============================================================
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}

DONE_TOOL = {
    "name": "return_job_listings",
    "description": (
        "Call this when you have finished searching and have a complete list "
        "of relevant job postings. Pass the structured list of jobs as the argument."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":   {"type": "string"},
                        "company": {"type": "string"},
                        "url":     {"type": "string"},
                    },
                    "required": ["title", "company", "url"],
                },
            },
        },
        "required": ["jobs"],
    },
}


# ============================================================
# STEP 3: THE SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """You are a job search agent finding early-career engineering roles.

Your task:
1. Search for job postings across: company career pages, Wellfound, YC Work at a Startup,
   greenhouse.io, lever.co, and LinkedIn posts where hiring managers posted directly
   (look for "we're hiring", "join our team", "DM me", "apply below").
2. Avoid: Indeed, ZipRecruiter, Glassdoor, Monster, Simply Hired.
3. Target roles in: agentic AI, AI voice agents, full stack (React/Node.js), backend (Python/FastAPI/Node.js).
4. Only include roles requiring 0-2 years of experience. Exclude anything asking for 4+ years,
   pure ML research, data science, non-technical AI roles, hardware engineering,
   embedded systems, firmware, FPGA, electrical engineering, or any physical/chip design roles. Exclude jobs that explicitly mention citizenship requirements or security clearance.
5. Run exactly 1 search, no more.
6. Return only job title, company, and URL — no summaries or explanations.

Today's date: """ + datetime.now().strftime("%B %d, %Y")


# ============================================================
# STEP 4: THE AGENT LOOP
# ============================================================

def run_job_search_agent() -> dict:
    """Run the ReAct agent loop and return structured job listings."""

    print("🤖 Starting job search agent...")

    messages = [
        {
            "role": "user",
            "content": (
                "Search for early-career agentic AI, voice AI, full stack, and backend "
                "engineering job postings. Run exactly 1 search, then call "
                "return_job_listings with title, company, and URL only."
            ),
        }
    ]

    tools = [WEB_SEARCH_TOOL, DONE_TOOL]
    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Agent iteration {iteration}...")

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        print(f"   Stop reason: {response.stop_reason}")

        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    if block.text.strip():
                        print(f"   💭 Claude: {block.text[:200]}...")

                elif block.type == "tool_use":
                    print(f"   🔧 Tool call: {block.name}")

                    if block.name == "return_job_listings":
                        print(f"   ✅ Agent finished! Found {len(block.input.get('jobs', []))} jobs.")
                        return block.input

                    elif block.name == "web_search":
                        print(f"   🔍 Searching: {block.input.get('query', '')}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            print("   ⚠️  Agent ended without calling return_job_listings")
            break

    return {"jobs": []}


# ============================================================
# STEP 5: PRINT THE DIGEST
# ============================================================

def print_digest(job_data: dict):
    """Print the job digest to the terminal."""
    jobs = job_data.get("jobs", [])
    date_str = datetime.now().strftime("%b %d, %Y")

    print("\n" + "=" * 60)
    print(f"  🤖 JOB DIGEST — {date_str}")
    print("=" * 60)
    print(f"  {len(jobs)} roles found\n")

    for job in jobs:
        print(f"  {job.get('title', '?')} @ {job.get('company', '?')}")
        print(f"  {job.get('url', '')}\n")


def run():
    """Run one cycle of the job monitor agent."""
    print("=" * 60)
    print("JOB MONITOR AGENT")
    print("=" * 60)

    job_data = run_job_search_agent()

    if not job_data.get("jobs"):
        print("No jobs found. Check your search terms or API connectivity.")
        return

    print_digest(job_data)


if __name__ == "__main__":
    run()