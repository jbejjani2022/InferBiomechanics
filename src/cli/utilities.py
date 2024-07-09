import subprocess


# Utility to get the current repo's git hash, which is useful for replicating runs later
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Git hash could not be found."
    
# Utility to check if the current repo has uncommited changes, which is useful for debugging why we can't replicate
# runs later, and also yelling at people if they run experiments with uncommited changes.
def has_uncommitted_changes():
    try:
        # The command below checks for changes including untracked files.
        # You can modify this command as per your requirement.
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        return bool(status)
    except subprocess.CalledProcessError:
        return "Could not determine if there are uncommitted changes."