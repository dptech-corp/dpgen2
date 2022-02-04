import sys, subprocess

def run_command(
        cmd,
        shell = None,
):
    pp = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        shell=shell,
    )
    out, err = pp.communicate()
    return_code = pp.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    return return_code, out, err
    
