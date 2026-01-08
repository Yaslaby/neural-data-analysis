import os
import numpy as np
from MultiChannelLoader import load_continuous_with_timestamps

# Test file path
file_path = "/Users/yaslaby/Downloads/100_1.continuous"

result = load_continuous_with_timestamps(file_path, verbose=True)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"File: {os.path.basename(file_path)}")
print(f"Samples loaded: {len(result['data'])}")
print(f"Records loaded: {result['record_count']}")
print(f"Block length: {result['header'].get('blockLength', 'N/A')}")
print(f"Expected total: {result['record_count'] * result['header'].get('blockLength', 0)}")
print(f"Difference: {abs(len(result['data']) - (result['record_count'] * result['header'].get('blockLength', 0)))}")
print("="*60)