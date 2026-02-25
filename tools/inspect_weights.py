import pickle
import numpy as np
import sys
import argparse

def inspect(pkl_path, search_term=None):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total keys: {len(data)}")
    
    count = 0
    for k, v in data.items():
        if search_term and search_term not in k:
            continue
        print(f"{k}: {v.shape}")
        count += 1
        if count >= 100: # Limit output
            print("... (truncated)")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--search', type=str, default=None)
    args = parser.parse_args()
    
    inspect(args.path, args.search)
