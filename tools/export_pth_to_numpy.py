
import torch
import pickle
import numpy as np
import sys
import os

def export_pth(pth_path, output_path):
    print(f"Loading {pth_path}...")
    try:
        # Load on CPU
        checkpoint = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading pth: {e}")
        return

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Loaded state_dict with {len(state_dict)} keys.")
    
    # Convert to numpy
    numpy_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            numpy_dict[k] = v.cpu().numpy()
        else:
            numpy_dict[k] = v
            
    # Save as pickle
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(numpy_dict, f)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_pth_to_numpy.py <pth_path> <output_path>")
        sys.exit(1)
        
    export_pth(sys.argv[1], sys.argv[2])
