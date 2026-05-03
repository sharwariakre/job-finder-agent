"""
Job Monitor Agent
=================
This is a ReAct-style AI agent that:
1. Uses Claude + web_search to find agentic AI job postings
2. Filters/ranks them with an LLM pass
3. Emails you a daily digest

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


# Job search configuration
JOB_SEARCHES = [
    "agentic AI engineer jobs 2025",
    "AI agent software engineer jobs",
    "LLM applications engineer job posting",
    "autonomous AI systems engineer careers",
]

# ============================================================
# STEP 1: THE ANTHROPIC CLIENT
# ============================================================
# The client connects to the Anthropic API.
# Your API key is read from the ANTHROPIC_API_KEY environment variable.
# Set it: export ANTHROPIC_API_KEY="sk-ant-..."
client = anthropic.Anthropic()


# ============================================================
# STEP 2: TOOL DEFINITIONS
# ============================================================
# We define tools as JSON schemas. Claude reads these schemas and
# decides WHEN and HOW to call each tool. We never hardcode "call
# web_search now" — Claude decides based on reasoning.

# The web_search tool is built into the Anthropic API.
# We just declare we want it available.
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}

# We also define a custom "done" tool. This is the signal Claude uses
# to exit the loop cleanly with structured output (the job listings).
# Without this, Claude might keep searching indefinitely.
DONE_TOOL = {
    "name": "return_job_listings",
    "description": (
        "Call this when you have finished searching and have a complete list "
        "of relevant agentic AI / AI engineer job postings. Pass the structured "
        "list of jobs as the argument."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":       {"type": "string"},
                        "company":     {"type": "string"},
                        "location":    {"type": "string"},
                        "url":         {"type": "string"},
                        "why_relevant":{"type": "string"},  # LLM explains why this fits
                        "seniority":   {"type": "string", "enum": ["entry", "mid", "senior", "staff", "unknown"]},
                    },
                    "required": ["title", "company", "url", "why_relevant"],
                },
            },
            "search_summary": {
                "type": "string",
                "description": "Brief summary of what you searched for and how many results you found.",
            },
        },
        "required": ["jobs", "search_summary"],
    },
}


# ============================================================
# STEP 3: THE SYSTEM PROMPT
# ============================================================
# The system prompt is Claude's "personality" and instructions for this task.
# It shapes how Claude reasons inside the ReAct loop.

SYSTEM_PROMPT = """You are a job search agent specialized in finding agentic AI and LLM engineering roles.

Your task:
1. Search the web for recent job postings for agentic AI engineers, LLM application engineers, AI systems engineers, 
   and similar roles that involve building autonomous AI agents, LLM pipelines, or AI-powered applications.
2. Filter results: only keep roles that involve actually BUILDING AI agents/systems (not just using ChatGPT at work).
3. Look for signals like: ReAct loops, tool use, LLM orchestration, agent frameworks (LangChain, LlamaIndex, 
   AutoGen, CrewAI), RAG systems, voice AI, multimodal AI, or autonomous systems.
4. Exclude: data science roles focused on training models, pure ML research, non-technical AI roles.
5. Run 3-4 searches with different query terms to maximize coverage.
6. Once you have a solid list (aim for 10-20 relevant jobs), call return_job_listings with your findings.

For each job, explain WHY it's relevant to agentic AI work specifically.
Today's date: """ + datetime.now().strftime("%B %d, %Y")


# ============================================================
# STEP 4: THE AGENT LOOP
# ============================================================

def run_job_search_agent() -> dict:
    """
    Run the ReAct agent loop.
    
    Returns the structured job listings from Claude's return_job_listings call.
    
    The loop:
      messages = [initial user request]
      while True:
          response = claude(messages, tools=[web_search, return_job_listings])
          
          if stop_reason == "end_turn":
              break  # Claude finished without calling return_job_listings
          
          for block in response.content:
              if block.type == "tool_use":
                  if block.name == "return_job_listings":
                      return block.input   # We're done! Claude gave us structured output.
                  elif block.name == "web_search":
                      # The API handles web_search execution automatically.
                      # We don't need to execute it ourselves — the results
                      # come back in the next response as tool_result blocks.
                      pass
          
          # Append assistant response to message history so Claude remembers what it did
          messages.append({"role": "assistant", "content": response.content})
          # Continue loop — Claude will react to search results
    """
    
    print("🤖 Starting job search agent...")
    
    # The conversation starts with a single user message.
    # Everything Claude does from here is autonomous.
    messages = [
        {
            "role": "user",
            "content": (
                "Please search for agentic AI and LLM engineer job postings. "
                "Search multiple queries, filter for genuine agent/LLM roles, "
                "then call return_job_listings with your findings."
            ),
        }
    ]
    
    tools = [WEB_SEARCH_TOOL, DONE_TOOL]
    iteration = 0
    max_iterations = 10  # Safety limit — agents can get stuck in loops
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Agent iteration {iteration}...")
        
        # THE CORE API CALL
        # This is where Claude thinks, decides which tool to call, and returns.
        # Note: betas=["interleaved-thinking-2025-05-14"] enables extended thinking
        # which lets Claude reason more carefully before acting.
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        
        print(f"   Stop reason: {response.stop_reason}")
        
        # INSPECT THE RESPONSE
        # response.content is a list of blocks. Each block is one of:
        #   - TextBlock: Claude's reasoning/commentary (we log this)
        #   - ToolUseBlock: Claude wants to call a tool
        #   - ToolResultBlock: results from a tool (for web_search, API handles this)
        
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    # Claude is thinking out loud (ReAct = Reasoning step)
                    if block.text.strip():
                        print(f"   💭 Claude: {block.text[:200]}...")
                
                elif block.type == "tool_use":
                    print(f"   🔧 Tool call: {block.name}")
                    
                    # CHECK FOR OUR DONE SIGNAL
                    if block.name == "return_job_listings":
                        print(f"   ✅ Agent finished! Found {len(block.input.get('jobs', []))} jobs.")
                        return block.input  # Return the structured data
                    
                    elif block.name == "web_search":
                        print(f"   🔍 Searching: {block.input.get('query', '')}")
                        # The web_search tool is handled automatically by the Anthropic API.
                        # We don't execute it — the next response will contain the results.
        
        # If Claude didn't call return_job_listings yet, continue the loop.
        # Append the response so Claude has memory of what it did.
        messages.append({"role": "assistant", "content": response.content})
        
        # If Claude is done without calling our tool, break
        if response.stop_reason == "end_turn":
            print("   ⚠️  Agent ended without calling return_job_listings")
            break
    
    # Fallback if agent loop exhausted
    return {"jobs": [], "search_summary": "Agent loop exhausted without results."}


# ============================================================
# ============================================================
# STEP 5: PRINT THE DIGEST
# ============================================================

def print_digest(job_data: dict):
    """Print the job digest to the terminal."""
    jobs = job_data.get("jobs", [])
    summary = job_data.get("search_summary", "")
    date_str = datetime.now().strftime("%b %d, %Y")

    print("\n" + "=" * 60)
    print(f"  🤖 AGENTIC AI JOBS — {date_str}")
    print("=" * 60)
    print(f"  {summary}")
    print(f"  {len(jobs)} relevant roles found\n")

    # Group by seniority for cleaner output
    groups = {
        "senior/staff": [j for j in jobs if j.get("seniority") in ("senior", "staff")],
        "mid":          [j for j in jobs if j.get("seniority") == "mid"],
        "other":        [j for j in jobs if j.get("seniority") not in ("senior", "staff", "mid")],
    }

    for label, group in groups.items():
        if not group:
            continue
        print(f"── {label.upper()} {'─' * (50 - len(label))}")
        for job in group:
            print(f"\n  {job.get('title', '?')}")
            print(f"  {job.get('company', '?')}  ·  {job.get('location', 'Remote/Unknown')}")
            print(f"  {job.get('url', '')}")
            print(f"  ↳ {job.get('why_relevant', '')}")
        print()


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