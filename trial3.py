import struct
import os
import glob

def count_samples_in_file(filepath):
    """Count total number of samples in a .continuous file"""
    total_samples = 0
    
    with open(filepath, 'rb') as f:
        # Skip 1024-byte header
        f.seek(1024)
        
        while True:
            # Read timestamp (8 bytes)
            timestamp_data = f.read(8)
            if len(timestamp_data) < 8:
                break
            
            # Read sample count (2 bytes)
            sample_count_data = f.read(2)
            if len(sample_count_data) < 2:
                break
            
            N = struct.unpack('<H', sample_count_data)[0]
            total_samples += N
            
            # Skip the rest of this block:
            # - recording number (2 bytes)
            # - samples (N * 2 bytes)
            # - record marker (10 bytes)
            f.seek(2 + N * 2 + 10, 1)
    
    return total_samples

def find_all_continuous_files(base_path):
    """Find all .continuous files in the directory"""
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Path does not exist: {base_path}")
        return []
    
    print(f"SEARCHING FOR .continuous FILES IN: {base_path}")
    print("-" * 60)
    
    # Search in current directory
    pattern = os.path.join(base_path, "*.continuous")
    files = glob.glob(pattern)
    
    # Also search recursively in subdirectories
    pattern_recursive = os.path.join(base_path, "**", "*.continuous")
    files_recursive = glob.glob(pattern_recursive, recursive=True)
    
    print(f"Found {len(files)} .continuous file(s) in main directory")
    print(f"Found {len(files_recursive)} .continuous file(s) including subdirectories")
    
    # Show what files ARE in the directory
    if len(files_recursive) == 0:
        print("\nLet me check what files are in this directory:")
        try:
            all_files = os.listdir(base_path)
            if all_files:
                print(f"Files/folders in {base_path}:")
                for item in sorted(all_files)[:20]:  # Show first 20
                    print(f"  - {item}")
                if len(all_files) > 20:
                    print(f"  ... and {len(all_files) - 20} more")
            else:
                print("Directory is empty!")
        except Exception as e:
            print(f"Could not list directory: {e}")
    
    print()
    return files_recursive if files_recursive else files

def test_all_files(base_path):
    """Count samples in all .continuous files"""
    
    # Find all continuous files
    files = find_all_continuous_files(base_path)
    
    if not files:
        print("No .continuous files found!")
        return {}
    
    results = {}
    
    print("=" * 80)
    print("COUNTING SAMPLES")
    print("=" * 80)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        
        try:
            print(f"\nCounting {filename}...")
            sample_count = count_samples_in_file(filepath)
            
            # Calculate duration at 20kHz sampling rate
            duration_seconds = sample_count / 20000
            duration_minutes = duration_seconds / 60
            
            results[filename] = {
                'filepath': filepath,
                'sample_count': sample_count,
                'duration_seconds': duration_seconds,
                'duration_minutes': duration_minutes
            }
            
            print(f"✓ {filename}: {sample_count:,} samples ({duration_minutes:.2f} minutes)")
            
        except Exception as e:
            print(f"✗ {filename}: ERROR - {str(e)}")
            results[filename] = {'error': str(e)}
    
    return results

def print_summary(results):
    """Print summary of sample counts"""
    
    print("\n" + "=" * 90)
    print("SAMPLE COUNT SUMMARY")
    print("=" * 90)
    
    # Filter out errors
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results found!")
        return
    
    print(f"{'Filename':<40} {'Sample Count':<20} {'Duration (min)':<15}")
    print("-" * 90)
    
    for filename, data in valid_results.items():
        print(f"{filename:<40} {data['sample_count']:>15,}     {data['duration_minutes']:>10.2f}")
    
    # Total statistics
    total_samples = sum(v['sample_count'] for v in valid_results.values())
    total_minutes = sum(v['duration_minutes'] for v in valid_results.values())
    
    print("-" * 90)
    print(f"{'TOTAL:':<40} {total_samples:>15,}     {total_minutes:>10.2f}")
    
    # Show any errors
    error_results = {k: v for k, v in results.items() if 'error' in v}
    if error_results:
        print("\n" + "=" * 50)
        print("ERRORS:")
        print("=" * 50)
        for filename, data in error_results.items():
            print(f"{filename}: {data['error']}")

# MAIN EXECUTION
if __name__ == "__main__":
    # Set your base path where .continuous files are located
    base_path = "/Users/yaslaby/Downloads" \
    
    # First, let's explore the directory structure
    print("EXPLORING DIRECTORY STRUCTURE:")
    print("=" * 60)
    
    if os.path.exists(base_path):
        print(f"✓ Path exists: {base_path}\n")
        
        # List all files with their extensions
        print("All files in this directory:")
        for root, dirs, files in os.walk(base_path):
            level = root.replace(base_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                print(f'{subindent}{file}')
            if len(files) > 10:
                print(f'{subindent}... and {len(files) - 10} more files')
            if level > 2:  # Don't go too deep
                break
    else:
        print(f"✗ Path does NOT exist: {base_path}")
        
        # Try to find where the files might be
        parent = os.path.dirname(base_path)
        if os.path.exists(parent):
            print(f"\nParent directory exists: {parent}")
            print("Contents:")
            for item in os.listdir(parent)[:20]:
                print(f"  - {item}")
    
    print("\n" + "=" * 60)
    
    # Count samples in all continuous files
    results = test_all_files(base_path)
    
    # Print summary
    if results:
        print_summary(results)