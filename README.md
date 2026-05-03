# Job Monitor Agent

An agentic AI job search tool that uses Claude + web search to find agentic AI engineer roles and email you a daily digest.

## Setup

### 1. Install dependencies
```bash
pip install anthropic schedule
```

### 2. Set environment variables
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run it

```bash
# Test run (single execution, prints results, no schedule)
python scheduler.py --once

# Run on a schedule (every 6 hours)
python scheduler.py

# Custom interval
python scheduler.py --interval-hours 12
```

## How it works (the ReAct loop)

```
User message → Claude thinks → Claude calls web_search → sees results
→ Claude thinks again → Claude calls web_search again → sees results
→ Claude decides it has enough → Claude calls return_job_listings
→ We receive structured JSON → format HTML email → send via Gmail
```

This is the same loop Claude Code used when it applied to Medfinder autonomously.
The key insight: **Claude decides what to search and when to stop — we don't hardcode it.**

## Files

- `agent.py` — the agent loop, tool definitions, email formatting
- `scheduler.py` — runs agent on a schedule using the `schedule` library