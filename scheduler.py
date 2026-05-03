"""
scheduler.py
============
Runs the job monitor agent on a schedule.

Usage:
    python scheduler.py          # runs every 6 hours
    python scheduler.py --once   # run once and exit (useful for testing)

The schedule library is dead simple:
    schedule.every(6).hours.do(run)  # register the job
    while True:
        schedule.run_pending()        # check if any jobs are due
        time.sleep(60)                # check every minute
"""

import schedule
import time
import argparse
from agent import run

def main():
    parser = argparse.ArgumentParser(description="Job Monitor Agent Scheduler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval-hours", type=int, default=6, help="Hours between runs")
    args = parser.parse_args()
    
    if args.once:
        print("Running once...")
        run()
        return
    
    # Schedule the job
    schedule.every(args.interval_hours).hours.do(run)
    
    # Run immediately on startup too
    print(f"🕐 Job monitor started. Running now, then every {args.interval_hours} hours.")
    run()
    
    # The scheduler loop
    # This is NOT an agent loop — it's just a cron-like timer.
    # The agent loop lives inside agent.run_job_search_agent()
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute if a scheduled job is due

if __name__ == "__main__":
    main()