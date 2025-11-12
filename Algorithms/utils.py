import subprocess


def get_job_state(job_id):
    # First try squeue (for active jobs)
    squeue = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True, text=True
    )
    state = squeue.stdout.strip()
    if state:
        return state  # e.g. "RUNNING", "PENDING"

    # If not in queue, check sacct (for finished jobs)
    sacct = subprocess.run(
        ["sacct", "-j", str(job_id), "--noheader", "--format=State"],
        capture_output=True, text=True
    )
    #lines = [l.strip() for l in sacct.stdout.splitlines() if l.strip()]
    #if not lines:
    #    return "UNKNOWN"  # maybe DB delay
    #return lines[-1]  # usually the main job

    main_state = None
    for line in sacct.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        jobid = parts[0]
        state = parts[1] if len(parts) > 1 else "UNKNOWN"
        if jobid == str(job_id):
            main_state = state
            break  # we found the main job, no need to continue
    return main_state or "UNKNOWN"
