import os
import numpy as np
from MultiChannelLoader import load_continuous_with_timestamps
from integrated_mne_processing import process_for_ripples_mne_standard

# File path
file_path = "/Users/yaslaby/Documents/PyQt5_project/Channels/120_CH101.continuous"

print("="*70)
print("TRACKING SAMPLE COUNT DISCREPANCY")
print("="*70)

try:
    # Step 1: Load original data
    print("\n📁 STEP 1: LOADING ORIGINAL FILE")
    print("-" * 70)
    result = load_continuous_with_timestamps(file_path, verbose=True)
    
    original_data = result['data']
    original_fs = result['sample_rate']
    record_count = result['record_count']
    block_length = result['header'].get('blockLength', 1024)
    
    print(f"\n✓ Original data loaded:")
    print(f"  Samples: {len(original_data):,}")
    print(f"  Sample rate: {original_fs} Hz")
    print(f"  Records: {record_count:,}")
    print(f"  Block length: {block_length}")
    print(f"  Duration: {len(original_data) / original_fs:.2f} seconds")
    
    # Calculate what supervisor should get
    print(f"\n📊 SUPERVISOR'S NUMBERS:")
    supervisor_downsampled = 1809971
    supervisor_duration = supervisor_downsampled / 1000
    supervisor_original = int(supervisor_downsampled * (original_fs / 1000))
    
    print(f"  Expected original samples @ {original_fs} Hz: {supervisor_original:,}")
    print(f"  Expected downsampled @ 1000 Hz: {supervisor_downsampled:,}")
    print(f"  Expected duration: {supervisor_duration:.2f} seconds")
    
    # Show the difference
    sample_difference = supervisor_original - len(original_data)
    print(f"\n⚠️  DIFFERENCE IN ORIGINAL DATA:")
    print(f"  Your samples: {len(original_data):,}")
    print(f"  Supervisor's expected: {supervisor_original:,}")
    print(f"  Missing: {sample_difference:,} samples")
    print(f"  Missing duration: {sample_difference / original_fs:.2f} seconds")
    print(f"  Missing records: {sample_difference / block_length:.1f} records")
    
    # Step 2: Downsample
    print(f"\n🔽 STEP 2: DOWNSAMPLING TO 1000 Hz")
    print("-" * 70)
    
    mne_results = process_for_ripples_mne_standard(
        original_data, 
        fs_orig=original_fs, 
        fs_target=1000,
        show_plot=False
    )
    
    downsampled_data = mne_results['downsampled']
    
    print(f"✓ Downsampled:")
    print(f"  Samples: {len(downsampled_data):,}")
    print(f"  Duration: {len(downsampled_data) / 1000:.2f} seconds")
    
    downsampled_difference = supervisor_downsampled - len(downsampled_data)
    print(f"\n⚠️  DIFFERENCE AFTER DOWNSAMPLING:")
    print(f"  Your samples: {len(downsampled_data):,}")
    print(f"  Supervisor's: {supervisor_downsampled:,}")
    print(f"  Missing: {downsampled_difference:,} samples")
    print(f"  Missing duration: {downsampled_difference / 1000:.2f} seconds")
    
    # Step 3: Check file integrity
    print(f"\n🔍 STEP 3: FILE INTEGRITY CHECK")
    print("-" * 70)
    
    # Get actual file size
    file_size = os.path.getsize(file_path)
    header_size = 1024
    record_size = 2 * block_length + 22
    
    # Calculate theoretical record count
    data_size = file_size - header_size
    theoretical_records = data_size // record_size
    theoretical_samples = theoretical_records * block_length
    
    print(f"File analysis:")
    print(f"  Total file size: {file_size:,} bytes")
    print(f"  Header size: {header_size:,} bytes")
    print(f"  Data size: {data_size:,} bytes")
    print(f"  Record size: {record_size:,} bytes per record")
    print(f"  Theoretical max records: {theoretical_records:,}")
    print(f"  Theoretical max samples: {theoretical_samples:,}")
    print(f"  Actual records loaded: {record_count:,}")
    print(f"  Actual samples loaded: {len(original_data):,}")
    
    records_not_loaded = theoretical_records - record_count
    samples_not_loaded = theoretical_samples - len(original_data)
    
    if records_not_loaded > 0:
        print(f"\n⚠️  INCOMPLETE FILE READING:")
        print(f"  Records not loaded: {records_not_loaded:,}")
        print(f"  Samples not loaded: {samples_not_loaded:,}")
        print(f"  Duration not loaded: {samples_not_loaded / original_fs:.2f} seconds")
        print(f"\n💡 This might be why your numbers don't match supervisor's!")
    
    # Step 4: Check what's at the end of the file
    print(f"\n🔚 STEP 4: CHECKING END OF FILE")
    print("-" * 70)
    
    with open(file_path, 'rb') as f:
        f.seek(0, 2)  # Go to end
        file_end = f.tell()
        
        # Try to read the last record
        last_record_start = header_size + (record_count * record_size)
        remaining_bytes = file_end - last_record_start
        
        print(f"File ends at byte: {file_end:,}")
        print(f"Last record loaded at byte: {last_record_start:,}")
        print(f"Remaining bytes: {remaining_bytes:,}")
        print(f"Remaining records possible: {remaining_bytes / record_size:.2f}")
        
        if remaining_bytes >= record_size:
            print(f"\n⚠️  WARNING: There are {int(remaining_bytes / record_size)} complete records not loaded!")
            print(f"  This accounts for ~{int(remaining_bytes / record_size) * block_length:,} samples")
            print(f"  Duration: {int(remaining_bytes / record_size) * block_length / original_fs:.2f} seconds")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL ANALYSIS")
    print("="*70)
    print(f"\n{'Stage':<35} {'Your Count':<15} {'Supervisor':<15} {'Difference'}")
    print("-" * 70)
    print(f"{'Original samples':<35} {len(original_data):>14,} {supervisor_original:>14,} {sample_difference:>14,}")
    print(f"{'Downsampled samples (1000 Hz)':<35} {len(downsampled_data):>14,} {supervisor_downsampled:>14,} {downsampled_difference:>14,}")
    print(f"{'Duration (seconds)':<35} {len(original_data)/original_fs:>14.2f} {supervisor_duration:>14.2f} {(supervisor_duration - len(original_data)/original_fs):>14.2f}")
    
    print(f"\n{'='*70}")
    print("POTENTIAL CAUSES:")
    print("="*70)
    
    if records_not_loaded > 0:
        print(f"1. ⚠️  INCOMPLETE FILE READING")
        print(f"   → {records_not_loaded:,} records exist but weren't loaded")
        print(f"   → Check the loading loop termination condition")
        print(f"   → Line in MultiChannelLoader.py: 'while f.tell() < file_size - record_size'")
    
    if remaining_bytes >= record_size:
        print(f"\n2. ⚠️  PREMATURE LOOP TERMINATION")
        print(f"   → {int(remaining_bytes / record_size)} complete records at end of file ignored")
        print(f"   → The loader stops too early!")
    
    print(f"\n3. 💡 POSSIBLE FIX:")
    print(f"   Change line ~80 in MultiChannelLoader.py:")
    print(f"   FROM: while f.tell() < file_size - record_size:")
    print(f"   TO:   while f.tell() <= file_size - record_size:")
    print(f"   (Note: < changed to <=)")
    
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()