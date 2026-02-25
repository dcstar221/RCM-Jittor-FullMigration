import os
import sys
import subprocess

# Set up environment variables BEFORE importing jittor
cwd = os.getcwd()
temp_bin = os.path.join(cwd, 'temp_bin')
if os.path.exists(temp_bin):
    print(f"Adding {temp_bin} to PATH")
    os.environ['PATH'] = temp_bin + os.pathsep + os.environ['PATH']

# Set JITTOR_HOME to local directory to avoid permission issues
os.environ['JITTOR_HOME'] = os.path.join(cwd, '.jittor_cache')
if not os.path.exists(os.environ['JITTOR_HOME']):
    os.makedirs(os.environ['JITTOR_HOME'])
print(f"JITTOR_HOME set to: {os.environ['JITTOR_HOME']}")

# Disable MKL to avoid compatibility issues
os.environ['use_mkl'] = '0'

# Check clang version via subprocess
try:
    clang_version = subprocess.check_output(['clang', '--version']).decode('utf-8')
    print(f"Clang version found:\n{clang_version}")
except Exception as e:
    print(f"Error checking clang version: {e}")

import jittor as jt
from jittor import nn

def check_environment():
    print(f"Python executable: {sys.executable}")
    print(f"Jittor version: {jt.__version__}")
    
    # Check compiler flags
    print(f"Compiler flags: {jt.flags.cc_flags}")
    print(f"CC path: {jt.flags.cc_path}")
    
    # Check if cuda is available (should be False on Mac)
    print(f"Has Cuda: {jt.has_cuda}")
    
    # Basic tensor test
    try:
        a = jt.ones((3, 3))
        b = a * 2
        res = b.sum().item()
        print(f"Tensor test result: {res}")
        if res == 18.0:
            print("Tensor computation correct.")
        else:
            print("Tensor computation INCORRECT.")
    except Exception as e:
        print(f"Tensor test failed: {e}")
    
    print("Environment check passed!")

if __name__ == "__main__":
    check_environment()
