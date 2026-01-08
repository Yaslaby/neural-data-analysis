import struct
import numpy as np
from scipy import signal
import os
import glob

def test_continuous_file_with_downsample(filepath):
    """Get sample count, start time, and downsample from .continuous file"""
    total_samples = 0
    start_time_raw = None
    all_data = []
    
    with open(filepath, 'rb') as f:
        # Skip 1024-byte header
        f.seek(1024)
        
        while True:
            # Read timestamp (8 bytes)
            timestamp_data = f.read(8)
            if len(timestamp_data) < 8:
                break
            
            timestamp = struct.unpack('<q', timestamp_data)[0]
            if start_time_raw is None:
                start_time_raw = timestamp
            
            # Read sample count (2 bytes)
            sample_count_data = f.read(2)
            if len(sample_count_data) < 2:
                break
            
            N = struct.unpack('<H', sample_count_data)[0]
            total_samples += N
            
            # Skip recording number (2 bytes)
            f.read(2)
            
            # Read actual samples (N * 2 bytes)
            samples_data = f.read(N * 2)
            samples = struct.unpack('>' + 'h' * N, samples_data)
            all_data.extend(samples)
            
            # Skip record marker (10 bytes)
            f.read(10)
    
    # Convert to numpy array and microvolts
    original_data = np.array(all_data, dtype=np.float32) * 0.195
    
    # Downsample from 20kHz to 1kHz (factor of 20)
    downsample_factor = 20
    downsampled_data = signal.decimate(original_data, downsample_factor)
    
    # Convert timestamp to seconds
    sampling_rate = 20000
    start_time_seconds = start_time_raw / sampling_rate
    
    return total_samples, start_time_raw, start_time_seconds, original_data, downsampled_data

def find_hpc_files(base_path):
    """Find all HPC channel files with flexible naming"""
    
    # Hippocampal channels from your table
    hpc_channels = {
        'Rat1': 'CH101',
        'Rat2': 'CH69', 
        'Rat3': 'CH59',
        'Rat4': 'AUX8',
        'Rat5': 'CH124',
        'Rat6': 'CH58',
        'Rat7': 'CH71',
        'Rat8': 'CH25'
    }
    
    found_files = {}
    
    print("SEARCHING FOR HPC CHANNEL FILES...")
    print("-" * 50)
    
    for rat, channel in hpc_channels.items():
        # Search for any file containing the channel number
        pattern = os.path.join(base_path, f"*_{channel}.continuous")
        matches = glob.glob(pattern)
        
        if matches:
            # Take the first match if multiple found
            filepath = matches[0]
            filename = os.path.basename(filepath)
            found_files[rat] = {
                'channel': channel,
                'filepath': filepath,
                'filename': filename
            }
            print(f"✓ {rat} {channel}: Found {filename}")
        else:
            print(f"✗ {rat} {channel}: NOT FOUND (pattern: *_{channel}.continuous)")
            found_files[rat] = {
                'channel': channel,
                'error': 'File not found'
            }
    
    return found_files

def test_all_hpc_channels_flexible(base_path):
    """Test all HPC channels with flexible file finding"""
    
    # First, find all the files
    file_info = find_hpc_files(base_path)
    
    results = {}
    
    print("\n" + "=" * 80)
    print("TESTING FOUND HIPPOCAMPAL CHANNELS")
    print("=" * 80)
    
    for rat, info in file_info.items():
        if 'error' in info:
            results[rat] = info
            continue
            
        try:
            print(f"\nTesting {rat} - {info['channel']} ({info['filename']})...")
            samples, raw_time, time_seconds, original, downsampled = test_continuous_file_with_downsample(info['filepath'])
            
            # Calculate session estimate (30min = 1800 seconds per session)
            session_estimate = int(time_seconds / 1800) + 1
            
            results[rat] = {
                'channel': info['channel'],
                'filename': info['filename'],
                'filepath': info['filepath'],
                'raw_timestamp': raw_time,
                'time_seconds': time_seconds,
                'time_minutes': time_seconds / 60,
                'time_hours': time_seconds / 3600,
                'session_estimate': session_estimate,
                'total_samples': len(original),
                'duration_seconds': len(original) / 20000,
                'downsampled_samples': len(downsampled)
            }
            
            print(f"✓ {rat} {info['channel']}: {raw_time} → {time_seconds/3600:.2f}h → Session ~{session_estimate}")
            
        except Exception as e:
            print(f"✗ {rat} {info['channel']}: ERROR - {str(e)}")
            results[rat] = {'channel': info['channel'], 'error': str(e)}
    
    return results

def print_hpc_summary(results):
    """Print summary of all hippocampal channel results"""
    
    print("\n" + "=" * 110)
    print("HIPPOCAMPAL CHANNELS TIMESTAMP SUMMARY")
    print("=" * 110)
    
    # Filter out errors
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results found!")
        return
    
    # Sort by timestamp
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['raw_timestamp'])
    
    print(f"{'Rat':<6} {'Channel':<8} {'Filename':<20} {'Raw Timestamp':<12} {'Hours':<8} {'Session':<8} {'Duration':<10}")
    print("-" * 110)
    
    for rat, data in sorted_results:
        print(f"{rat:<6} {data['channel']:<8} {data['filename']:<20} {data['raw_timestamp']:<12} "
              f"{data['time_hours']:<8.2f} {data['session_estimate']:<8} {data['duration_seconds']:<10.1f}")
    
    # Show timestamp progression
    print(f"\nTIMESTAMP PROGRESSION:")
    print("-" * 50)
    
    for i, (rat, data) in enumerate(sorted_results):
        if i == 0:
            print(f"{rat} {data['channel']}: {data['time_hours']:.2f}h (EARLIEST)")
        elif i == len(sorted_results) - 1:
            print(f"{rat} {data['channel']}: {data['time_hours']:.2f}h (LATEST)")
        else:
            print(f"{rat} {data['channel']}: {data['time_hours']:.2f}h")
    
    # Show time span
    earliest = sorted_results[0][1]['time_hours']
    latest = sorted_results[-1][1]['time_hours']
    print(f"\nTime span: {latest - earliest:.2f} hours ({(latest - earliest) * 2:.0f} sessions)")

# MAIN EXECUTION
if __name__ == "__main__":
    # Set your base path where .continuous files are located
    base_path = "/Users/yaslaby/Documents/PyQt5_project/Channels/120_CH110"  # Adjust this path
    
    # Test all hippocampal channels with flexible file finding
    hpc_results = test_all_hpc_channels_flexible(base_path)
    
    # Print comprehensive summary
    print_hpc_summary(hpc_results)
    
    # Show any errors
    print("\n" + "=" * 50)
    print("ERRORS:")
    print("=" * 50)
    for rat, data in hpc_results.items():
        if 'error' in data:
            print(f"{rat} {data['channel']}: {data['error']}")