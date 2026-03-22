import subprocess

# 
# TODO: fix the bug or remove this function
#
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


# checking squeue is not enough, sacct should be used to confirm completion
def is_job_completed(job_id):
    """
    Checks if a Slurm job has completed using squeue.
    Returns True if completed, False otherwise (running, pending, or not found).
    """
    try:
        # Execute squeue command for the specific job ID
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h"],  # -h for no header
            capture_output=True,
            text=True,
            check=True
        )
        # If the output is empty, the job is not in the queue (likely completed)
        return not bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        # This might happen if the job_id is invalid or other squeue issues
        print(f"Error checking job status with squeue: {e}")
        return False
    

def is_job_completed_sacct(job_id):
    """
    Checks if a Slurm job has completed using sacct.
    Returns True if completed, False otherwise.
    Only considers the exact job_id (ignores .batch, .extern, etc.).
    """
    try:
        # Run sacct with pipe-delimited format and no header
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=JobID,State", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True
        )

        output_lines = result.stdout.strip().split("\n")
        if not output_lines:
            return False  # Job not found in accounting records

        main_job_state = None
        for line in output_lines:
            fields = line.strip().split("|")
            if len(fields) < 2:
                continue
            jobid, state = fields[0], fields[1].strip()
            if jobid == str(job_id):  # exact match (no suffix)
                main_job_state = state
                break

        if main_job_state is None:
            return False  # No exact match found yet in sacct output

        # Common end states for finished jobs
        completed_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}

        # if main_job_state in completed_states:
        #     print(f"Job {job_id} state from sacct: {main_job_state}")
        return main_job_state in completed_states

    except subprocess.CalledProcessError as e:
        print(f"Error checking job status with sacct: {e}")
        print(f"stderr: {e.stderr}")
        return False
   
    