import sys
import os

# Set up environment variables BEFORE importing jittor
cwd = os.getcwd()
temp_bin = os.path.join(cwd, 'temp_bin')
if os.path.exists(temp_bin):
    os.environ['PATH'] = temp_bin + os.pathsep + os.environ['PATH']
    
# Set JITTOR_HOME to local directory to avoid permission issues
os.environ['JITTOR_HOME'] = os.path.join(cwd, '.jittor_cache')
if not os.path.exists(os.environ['JITTOR_HOME']):
    os.makedirs(os.environ['JITTOR_HOME'])
    
# Disable MKL
os.environ['use_mkl'] = '0'

# Add projects/mmdet3d_plugin to sys.path
sys.path.insert(0, os.path.join(cwd, 'RCM-Fusion-Jittor'))

try:
    from projects.mmdet3d_plugin.jittor_adapter import Config, Registry, build_from_cfg
    print("Successfully imported jittor_adapter")
    
    # Test registry
    TEST_REGISTRY = Registry('test')
    
    @TEST_REGISTRY.register_module()
    class TestModule:
        def __init__(self, a, b):
            self.a = a
            self.b = b
            print(f"TestModule initialized with a={a}, b={b}")
            
    cfg = {'type': 'TestModule', 'a': 1, 'b': 2}
    obj = build_from_cfg(cfg, TEST_REGISTRY)
    
    if isinstance(obj, TestModule) and obj.a == 1 and obj.b == 2:
        print("Registry and build_from_cfg test passed")
    else:
        print("Registry and build_from_cfg test FAILED")

except ImportError as e:
    print(f"Failed to import jittor_adapter: {e}")
except Exception as e:
    print(f"Test failed with error: {e}")
