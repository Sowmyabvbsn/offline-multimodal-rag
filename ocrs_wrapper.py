import subprocess
import sys


def run_ocrs_cli(args):
    """
    Run ocrs-cli.exe with the given arguments.
    Args:
        args (list): List of command-line arguments (excluding the executable).
    Returns:
        stdout, stderr: Output and error from the process.
    """
    exe = "ocrs-cli.exe"  # Change to "ocrs-clie.exe" or "ocrs.exe" as needed
    cmd = [exe] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

if __name__ == "__main__":
    # Example: python ocrs_wrapper.py --json cvr.pdf --pdf
    # All CLI args after the script name are passed to ocrs-cli.exe
    cli_args = sys.argv[1:]
    out, err = run_ocrs_cli(cli_args)
    print("Output:\n", out)
    if err:
        print("Error:\n", err)
