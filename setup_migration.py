#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import platform
import glob

def find_brew_prefix():
    try:
        prefix = subprocess.check_output(['brew', '--prefix']).decode('utf-8').strip()
        if os.path.isdir(prefix):
            return prefix
    except:
        pass
    if os.path.exists('/opt/homebrew'):
        return '/opt/homebrew'
    if os.path.exists('/usr/local'):
        return '/usr/local'
    return '/usr/local'

def find_llvm_clang(brew_prefix):
    candidates = [
        os.path.join(brew_prefix, 'opt/llvm/bin/clang'),
        os.path.join(brew_prefix, 'opt/llvm@16/bin/clang'),
        os.path.join(brew_prefix, 'opt/llvm@15/bin/clang'),
        os.path.join(brew_prefix, 'opt/llvm@14/bin/clang'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c, c.replace('clang', 'clang++')
    
    # Prefer /usr/bin/clang (Apple Clang) over generic clang in PATH
    if os.path.exists('/usr/bin/clang'):
        return '/usr/bin/clang', '/usr/bin/clang++'
    
    try:
        subprocess.check_call(['clang', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'clang', 'clang++'
    except:
        pass
    return 'clang', 'clang++'

def find_libomp(brew_prefix):
    candidates = [
        os.path.join(brew_prefix, 'opt/libomp'),
        os.path.join(brew_prefix, 'lib/libomp.dylib'), 
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
        if os.path.isfile(c):
            return os.path.dirname(os.path.dirname(c))
    return None

def patch_jittor_utils(site_packages):
    # Fix IndexError in jittor_utils/__init__.py get_version()
    utils_path = os.path.join(site_packages, 'jittor_utils', '__init__.py')
    if not os.path.exists(utils_path):
        print(f"Warning: jittor_utils not found at {utils_path}")
        return False
        
    print(f"Patching jittor_utils at: {utils_path}")
    with open(utils_path, 'r') as f:
        content = f.read()
        
    if "version = \"(16.0.0)\" # Fallback" in content:
        print("jittor_utils already patched")
        return True

    with open(utils_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    patched = False
    for line in lines:
        if 'version = "("+v[-3]+")"' in line:
            # Replace fragile parsing with safer one
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            base_indent = " " * indent
            inner_indent = " " * (indent + 4)
            
            new_lines.append(f'{base_indent}try:\n')
            new_lines.append(f'{inner_indent}version = "("+v[-3]+")"\n')
            new_lines.append(f'{base_indent}except:\n')
            new_lines.append(f'{inner_indent}version = "(16.0.0)" # Fallback\n')
            patched = True
        else:
            new_lines.append(line)
            
    if patched:
        with open(utils_path, 'w') as f:
            f.writelines(new_lines)
        print("Successfully patched jittor_utils")
    else:
        print("jittor_utils already patched or structure changed")
    return True

def patch_jittor_utils_compiler_path(site_packages):
    # Fix compiler path detection in jittor_utils/__init__.py
    utils_path = os.path.join(site_packages, 'jittor_utils', '__init__.py')
    if not os.path.exists(utils_path):
        return False
        
    print(f"Patching jittor_utils compiler path at: {utils_path}")
    with open(utils_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    patched = False
    for line in lines:
        if "cc_path = env_or_find('cc_path', 'clang++', silent=True)" in line:
            indent = len(line) - len(line.lstrip())
            base_indent = " " * indent
            # Use /usr/bin/clang++ as default on macOS if available
            new_lines.append(f"{base_indent}cc_path = env_or_find('cc_path', '/usr/bin/clang++', silent=True)\n")
            patched = True
        else:
            new_lines.append(line)
            
    if patched:
        with open(utils_path, 'w') as f:
            f.writelines(new_lines)
        print("Successfully patched jittor_utils compiler path")
    else:
        print("jittor_utils compiler path already patched or not found")
    return True

def patch_compiler_py(site_packages, conda_prefix, libomp_path=None):
    compiler_py_path = os.path.join(site_packages, 'jittor', 'compiler.py')
    if not os.path.exists(compiler_py_path):
        print(f"Error: compiler.py not found at {compiler_py_path}")
        return False
        
    print(f"Patching compiler.py at: {compiler_py_path}")
    
    with open(compiler_py_path, 'r') as f:
        lines = f.readlines()
    
    # 1. Clean up previous patches to allow re-patching
    clean_lines = []
    for line in lines:
        # Remove our specific cc_flags patch line if present
        if "dynamic_lookup" in line and "cc_flags +=" in line:
            continue
            
        # Remove our return apple-m1 patch
        if "return \"apple-m1\"" in line and line.strip() == 'return "apple-m1"':
            continue
            
        clean_lines.append(line)
    
    lines = clean_lines
    new_lines = []
    patched = False
    lib_path = os.path.join(conda_prefix, 'lib')
    
    omp_flags = f" -L{lib_path} "
    omp_link = "-lomp"
    
    conda_omp = os.path.join(lib_path, "libomp.dylib")
    if os.path.exists(conda_omp):
        omp_link = conda_omp
    elif libomp_path:
         omp_flags += f" -L{libomp_path}/lib "
         if os.path.exists(os.path.join(libomp_path, "lib", "libomp.dylib")):
             omp_link = os.path.join(libomp_path, "lib", "libomp.dylib")

    for i, line in enumerate(lines):
        # Patch 1: Remove -lstdc++
        if "-lstdc++" in line:
            line = line.replace("-lstdc++", "")
            patched = True
        
        # Patch 2: Add -fpermissive and undefined lookup for Darwin (ONLY TOP LEVEL)
        if "if platform.system() == 'Darwin':" in line:
            indent = len(line) - len(line.lstrip())
            # Only apply to top-level check (indent 0) to avoid UnboundLocalError in functions
            if indent == 0:
                new_lines.append(line)
                new_lines.append(f"    cc_flags += \" -undefined dynamic_lookup -fpermissive -Xpreprocessor -fopenmp {omp_link} {omp_flags} \"\n")
                patched = True
                continue
            else:
                new_lines.append(line)
                continue

        # Patch 3: Fix OpenMP linking
        if "cc_flags += \" -lomp \"" in line:
             # Calculate indentation
            indent = len(line) - len(line.lstrip())
            current_indent = " " * indent
            line = f"{current_indent}cc_flags += \" -Xpreprocessor -fopenmp {omp_link} {omp_flags} \"\n"
            patched = True

        # Patch 4: Fix check_clang_latest_supported_cpu for Apple Silicon
        if "def check_clang_latest_supported_cpu():" in line:
            new_lines.append(line)
            new_lines.append('    return "apple-m1"\n')
            patched = True
            continue

        new_lines.append(line)
        
    if patched:
        with open(compiler_py_path, 'w') as f:
            f.writelines(new_lines)
        print("Successfully patched compiler.py")
    else:
        print("compiler.py might already be patched")
    return True

def main():
    print("=== RCM-Fusion Migration Setup Script v3 ===")
    
    brew_prefix = find_brew_prefix()
    print(f"Detected Homebrew prefix: {brew_prefix}")
    
    clang_path, clangpp_path = find_llvm_clang(brew_prefix)
    print(f"Selected Clang: {clang_path}")
    
    libomp_path = find_libomp(brew_prefix)
    if libomp_path:
        print(f"Selected LibOMP: {libomp_path}")
    else:
        print("WARNING: LibOMP not found!")
    
    conda_prefix = sys.exec_prefix
    print(f"Current Conda Env: {conda_prefix}")
    
    # Locate site-packages manually to avoid import errors
    site_packages = os.path.join(conda_prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
    if not os.path.exists(site_packages):
        # Fallback for some conda configs
        site_packages = glob.glob(os.path.join(conda_prefix, 'lib', 'python*', 'site-packages'))[0]
        
    print(f"Target site-packages: {site_packages}")
    
    # 1. Patch jittor_utils first (blocking import)
    print("\n--- Patching jittor_utils ---")
    patch_jittor_utils(site_packages)
    patch_jittor_utils_compiler_path(site_packages)
    
    # 2. Patch Jittor compiler
    print("\n--- Patching Jittor Compiler ---")
    patch_compiler_py(site_packages, conda_prefix, libomp_path)
    
    # 3. Generate source_env.sh
    print("\n--- Generating source_env.sh ---")
    cwd = os.getcwd()
    jittor_home = os.path.join(cwd, '.jittor_cache')
    if not os.path.exists(jittor_home):
        os.makedirs(jittor_home)
    
    with open('source_env.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated by setup_migration.py\n")
        f.write(f"export CC=\"{clang_path}\"\n")
        f.write(f"export CXX=\"{clangpp_path}\"\n")
        f.write(f"export cc_path=\"{clangpp_path}\"\n")
        
        if libomp_path:
            f.write(f"export LIBRARY_PATH=\"{libomp_path}/lib:{conda_prefix}/lib:$LIBRARY_PATH\"\n")
            f.write(f"export LD_LIBRARY_PATH=\"{libomp_path}/lib:{conda_prefix}/lib:$LD_LIBRARY_PATH\"\n")
            f.write(f"export DYLD_LIBRARY_PATH=\"{libomp_path}/lib:{conda_prefix}/lib:$DYLD_LIBRARY_PATH\"\n")
            f.write(f"export CPATH=\"{libomp_path}/include:$CPATH\"\n")
        
        f.write(f"export JITTOR_HOME=\"{jittor_home}\"\n")
        f.write(f"export use_mkl=0\n")
        f.write(f"export PYTHONNOUSERSITE=1\n")
        f.write(f"export PYTHONPATH=\"{cwd}:$PYTHONPATH\"\n")
        
    print("source_env.sh created.")
    print("\n=== Setup Complete ===")

if __name__ == "__main__":
    main()
