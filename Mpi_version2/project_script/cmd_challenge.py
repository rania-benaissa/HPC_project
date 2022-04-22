import os
import subprocess

config = None 

def usage():
    print("[challenge]")
    print("    challenge help")
    print("    challenge generate easy")
    print("    challenge generate medium")
    print("    challenge generate hard")
    print("    challenge generate HPC")

def help(*args):
    print("This module generate (personalized) challenge matrices.")
    print()
    print("easy   : About 1 hour of sequential computation")
    print("medium : About 1 day of sequential computation")
    print("hard   : About 1 month of sequential computation. About 3GB on disk")
    print("HPC    : About 5 years of sequential computation. About 20GB on disk.")


def prep():
    CC = os.environ.get('CC', 'gcc')
    if 'CC' in os.environ:
        print(f"---> Using {CC}")
    print("---> checking gcc", end='', flush=True)
    # ensure gcc works
    result = subprocess.run(f'{CC} --version', shell=True, check=True, capture_output=True)
    if "clang" in result.stdout.decode():
        raise ValueError(f"Your {CC} is in fact clang (common on OS X). This will not work. Set the CC environmnent variable.")
    print(" [OK]")
    
    # ensure gen_uniform is compiled
    print(f"---> compiling {config.MYDIR}/gen_uniform.c", end='', flush=True)
    subprocess.run(f'{CC} {config.MYDIR}/gen_uniform.c -O3 -o {config.MYDIR}/gen_uniform', shell=True, check=True, capture_output=True)
    print(" [OK]")


def generate(difficulty, *args):
    seed = 42
    file = f'challenge_{difficulty}.mtx'
    common = f'--seed {seed} --matrix {file}'

    if difficulty == 'easy':
        args = '--nrows 65536 --ncols 65000 --prime 65537 --per-row 50'
    elif difficulty == 'medium':
        args = '--nrows 200000 --ncols 198381 --prime 65537 --per-row 131'
    elif difficulty == 'hard':
        args = '--nrows 700814 --ncols 700000 --prime 65537 --per-row 253'
    elif difficulty == 'HPC':
        args = '--nrows 4000000 --ncols 3999635 --prime 65537 --per-row 199'
    else:
        raise ValueError("Unknown difficulty")

    prep()
    print(f"---> generating {file}")

    subprocess.run(f'{config.MYDIR}/gen_uniform  {common} {args}', shell=True, check=True)
    
    print(f"---> all set")


def setup(main):
    global config
    config = main
    mine = {'help': help, None: help, 'generate': generate}
    main.COMMANDS['challenge'] = mine
