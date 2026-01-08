import os
import numpy as np
import struct

def analyze_continuous_header(file_path):
    file_path = "/Users/yaslaby/Documents/PyGt5_project/120_CH1.continuous"

    """Analyze the continuous file header more carefully"""
    
    print(f"Detailed analysis of: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'rb') as f:
        # Read first 1024 bytes (header)
        header_bytes = f.read(1024)
        
        print("First 64 bytes (hex):")
        for i in range(0, 64, 16):
            hex_str = ' '.join(f'{b:02x}' for b in header_bytes[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in header_bytes[i:i+16])
            print(f"{i:04x}: {hex_str:<48} {ascii_str}")
        
        print("\nTrying different header interpretations:")
        
        # Method 1: Standard Open Ephys format
        f.seek(0)
        try:
            format_str = f.read(4).decode('utf-8', errors='ignore')
            version = struct.unpack('<I', f.read(4))[0]  # Little endian
            sample_rate = struct.unpack('<I', f.read(4))[0]
            block_length = struct.unpack('<H', f.read(2))[0]
            buffer_size = struct.unpack('<H', f.read(2))[0]
            channel = struct.unpack('<H', f.read(2))[0]
            
            print(f"\nMethod 1 (Little Endian):")
            print(f"  Format: '{format_str}'")
            print(f"  Version: {version}")
            print(f"  Sample rate: {sample_rate}")
            print(f"  Block length: {block_length}")
            print(f"  Buffer size: {buffer_size}")
            print(f"  Channel: {channel}")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Big endian
        f.seek(0)
        try:
            format_str = f.read(4).decode('utf-8', errors='ignore')
            version = struct.unpack('>I', f.read(4))[0]  # Big endian
            sample_rate = struct.unpack('>I', f.read(4))[0]
            block_length = struct.unpack('>H', f.read(2))[0]
            buffer_size = struct.unpack('>H', f.read(2))[0]
            channel = struct.unpack('>H', f.read(2))[0]
            
            print(f"\nMethod 2 (Big Endian):")
            print(f"  Format: '{format_str}'")
            print(f"  Version: {version}")
            print(f"  Sample rate: {sample_rate}")
            print(f"  Block length: {block_length}")
            print(f"  Buffer size: {buffer_size}")
            print(f"  Channel: {channel}")
            
        except Exception as e:
            print(f"Method 2 failed: {e}")
        
        # Method 3: Look for reasonable values in the header
        f.seek(0)
        header_data = f.read(1024)
        
        print(f"\nSearching for reasonable sample rates in header:")
        common_rates = [20000, 25000, 30000, 40000, 44100, 48000]
        
        for i in range(0, len(header_data) - 4, 4):
            value = struct.unpack('<I', header_data[i:i+4])[0]
            if value in common_rates:
                print(f"  Found {value} Hz at position {i}")
        
        # Look for typical block lengths
        print(f"\nSearching for typical block lengths (512, 1024, 2048):")
        typical_blocks = [512, 1024, 2048]
        
        for i in range(0, len(header_data) - 2, 2):
            value = struct.unpack('<H', header_data[i:i+2])[0]
            if value in typical_blocks:
                print(f"  Found {value} samples at position {i}")

def load_continuous_robust(file_path):
    """Load continuous file with robust header parsing"""
    
    with open(file_path, 'rb') as f:
        # Skip to data section first to verify structure
        f.seek(1024)  # Standard header size
        
        # Try to read first data block
        timestamp_bytes = f.read(8)
        if len(timestamp_bytes) < 8:
            raise Exception("Cannot read timestamp")
        
        timestamp = struct.unpack('<Q', timestamp_bytes)[0]  # uint64
        
        n_samples_bytes = f.read(2)
        if len(n_samples_bytes) < 2:
            raise Exception("Cannot read sample count")
        
        n_samples = struct.unpack('<H', n_samples_bytes)[0]  # uint16
        
        print(f"First data block:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Samples per block: {n_samples}")
        
        # Use reasonable defaults based on your data
        header = {
            'format': 'head',
            'version': 0,
            'sample_rate': 30000,  # Common rate, adjust if known
            'block_length': n_samples,
            'buffer_size': n_samples,
            'channel': 1  # Single channel
        }
        
        # Reset to data start
        f.seek(1024)
        
        # Read multiple data blocks
        data_blocks = []
        timestamps = []
        block_count = 0
        max_blocks = 1000  # Limit for testing
        
        try:
            while block_count < max_blocks:
                # Read timestamp
                timestamp_bytes = f.read(8)
                if len(timestamp_bytes) < 8:
                    break
                
                timestamp = struct.unpack('<Q', timestamp_bytes)[0]
                timestamps.append(timestamp)
                
                # Read number of samples
                n_samples_bytes = f.read(2)
                if len(n_samples_bytes) < 2:
                    break
                
                n_samples = struct.unpack('<H', n_samples_bytes)[0]
                
                # Skip recording number
                rec_num_bytes = f.read(2)
                if len(rec_num_bytes) < 2:
                    break
                
                # Read data
                data_bytes = f.read(n_samples * 2)
                if len(data_bytes) < n_samples * 2:
                    break
                
                block_data = np.frombuffer(data_bytes, dtype=np.int16)
                data_blocks.append(block_data)
                
                # Skip marker
                marker_bytes = f.read(10)
                if len(marker_bytes) < 10:
                    break
                
                block_count += 1
                
                # Print progress
                if block_count % 100 == 0:
                    print(f"  Read {block_count} blocks...")
        
        except Exception as e:
            print(f"Stopped after {block_count} blocks: {e}")
            if block_count == 0:
                raise
        
        if not data_blocks:
            raise Exception("No data blocks read")
        
        # Combine data
        data = np.concatenate(data_blocks)
        timestamps_array = np.array(timestamps)
        
        # Convert to microvolts
        data_uv = data.astype(np.float32) * 0.195
        
        print(f"\nLoaded successfully:")
        print(f"  Total blocks: {block_count}")
        print(f"  Total samples: {len(data)}")
        print(f"  Data range: {np.min(data_uv):.2f} to {np.max(data_uv):.2f} µV")
        print(f"  Estimated duration: {len(data) / header['sample_rate']:.2f} seconds")
        
        return data_uv, timestamps_array, header

# Test your specific file
if __name__ == "__main__":
    file_path = "/Users/yaslaby/Documents/PyGt5_project/120_CH1.continuous"
    
    print("Step 1: Analyzing header structure")
    analyze_continuous_header(file_path)
    
    print("\n" + "="*60)
    print("Step 2: Loading data with robust method")
    
    try:
        data, timestamps, header = load_continuous_robust(file_path)
        print("\n✅ File loaded successfully!")
        print(f"Final data shape: {data.shape}")
        print(f"Sample statistics:")
        print(f"  Mean: {np.mean(data):.2f} µV")
        print(f"  Std: {np.std(data):.2f} µV")
        print(f"  Min: {np.min(data):.2f} µV")
        print(f"  Max: {np.max(data):.2f} µV")
        
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")