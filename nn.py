#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def explain_minus_3db():
    """
    Simple explanation of what -3 dB means
    """
    
    print("WHAT DOES -3 dB MEAN?")
    print("=" * 22)
    
    # PART 1: The math behind dB
    print("\n1. THE MATH (don't worry, it's simple)")
    print("-" * 40)
    
    print("dB = 20 × log10(output/input)")
    print("\nLet's see what different ratios give us:")
    
    ratios = [1.0, 0.707, 0.5, 0.316, 0.1, 0.01]
    names = ["No change", "Half power", "Half amplitude", "1/√10", "1/10 amplitude", "1/100 amplitude"]
    
    for ratio, name in zip(ratios, names):
        db_value = 20 * np.log10(ratio)
        print(f"  {ratio:5.3f} → {db_value:6.1f} dB  ({name})")
    
    print(f"\n📍 NOTICE: 0.707 gives exactly -3.0 dB")
    print(f"   This is because 0.707 ≈ 1/√2 ≈ 0.7071...")
    
    # PART 2: Why -3 dB is special
    print("\n2. WHY -3 dB IS SPECIAL")
    print("-" * 27)
    
    print("Power vs Amplitude:")
    print("  • Power ∝ (amplitude)²")
    print("  • So if amplitude drops to 0.707...")
    print("  • Power drops to (0.707)² = 0.5 = HALF POWER!")
    print("")
    print("That's why -3 dB is called the 'half-power point'")
    
    # PART 3: Visual demonstration
    print("\n3. VISUAL DEMONSTRATION")
    print("-" * 26)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show sine waves at different amplitudes
    t = np.linspace(0, 2, 1000)
    
    # Original signal
    signal_1 = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
    signal_0707 = 0.707 * np.sin(2 * np.pi * 2 * t)  # -3 dB version
    signal_05 = 0.5 * np.sin(2 * np.pi * 2 * t)    # -6 dB version
    signal_01 = 0.1 * np.sin(2 * np.pi * 2 * t)    # -20 dB version
    
    ax1.plot(t, signal_1, 'b-', linewidth=3, label='Original (0 dB)')
    ax1.plot(t, signal_0707, 'r--', linewidth=2, label='-3 dB (0.707×)')
    ax1.plot(t, signal_05, 'g--', linewidth=2, label='-6 dB (0.5×)')
    ax1.plot(t, signal_01, 'm--', linewidth=2, label='-20 dB (0.1×)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Sine Waves at Different dB Levels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Power comparison (area under the curve squared)
    power_1 = signal_1**2
    power_0707 = signal_0707**2
    
    ax2.plot(t, power_1, 'b-', linewidth=3, label='Original power')
    ax2.plot(t, power_0707, 'r-', linewidth=3, label='-3 dB power (half!)')
    ax2.fill_between(t, 0, power_1, alpha=0.3, color='blue')
    ax2.fill_between(t, 0, power_0707, alpha=0.3, color='red')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power')
    ax2.set_title('Power: Original vs -3 dB')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Bar chart of amplitude ratios
    db_levels = [0, -3, -6, -10, -20, -40]
    amplitudes = [10**(db/20) for db in db_levels]  # Convert dB back to linear
    
    bars = ax3.bar(range(len(db_levels)), amplitudes, 
                   color=['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    ax3.set_xlabel('dB Level')
    ax3.set_ylabel('Amplitude Ratio')
    ax3.set_title('Amplitude at Different dB Levels')
    ax3.set_xticks(range(len(db_levels)))
    ax3.set_xticklabels([f'{db} dB' for db in db_levels])
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, amp) in enumerate(zip(bars, amplitudes)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{amp:.3f}', ha='center', va='bottom')
        if i == 1:  # Highlight -3 dB
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    # Power ratios
    power_ratios = [amp**2 for amp in amplitudes]
    bars2 = ax4.bar(range(len(db_levels)), power_ratios,
                    color=['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    ax4.set_xlabel('dB Level')
    ax4.set_ylabel('Power Ratio')
    ax4.set_title('Power at Different dB Levels')
    ax4.set_xticks(range(len(db_levels)))
    ax4.set_xticklabels([f'{db} dB' for db in db_levels])
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, power) in enumerate(zip(bars2, power_ratios)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{power:.3f}', ha='center', va='bottom')
        if i == 1:  # Highlight -3 dB
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    plt.tight_layout()
    plt.show()
    
    # PART 4: Real-world examples
    print("\n4. REAL-WORLD EXAMPLES")
    print("-" * 25)
    
    print("Audio/Music:")
    print("  • Your stereo volume knob: each major tick ≈ 3 dB")
    print("  • -3 dB = noticeably quieter but still clear")
    print("  • -20 dB = very quiet background music")
    
    print("\nIn your LFP filter:")
    print("  • 0 dB = frequencies pass through unchanged")
    print("  • -3 dB = this frequency gets reduced to 70.7%")
    print("  • -20 dB = this frequency gets reduced to 10%")
    print("  • -40 dB = this frequency gets reduced to 1%")
    
    # PART 5: Why engineers use -3 dB as cutoff
    print("\n5. WHY USE -3 dB AS 'CUTOFF'?")
    print("-" * 32)
    
    print("Historical reasons:")
    print("  • Easy to measure (half-power is obvious)")
    print("  • Good compromise (not too strict, not too loose)")
    print("  • Standard across all engineering fields")
    
    print("\nPractical reasons:")
    print("  • Still preserves most of the signal (70.7%)")
    print("  • Clear mathematical definition")
    print("  • Easy to calculate: just find where |H(f)| = 0.707")
    
    # PART 6: Back to your filter
    print("\n6. BACK TO YOUR MNE FILTER")
    print("-" * 28)
    
    print("What -3 dB cutoff at 500 Hz means:")
    print("  • Frequencies at 500 Hz get reduced to 70.7% amplitude")
    print("  • This is where your filter 'officially' starts cutting")
    print("  • Frequencies below 500 Hz: mostly preserved")
    print("  • Frequencies above 500 Hz: increasingly attenuated")
    
    print("\nFor your ripple analysis (150-250 Hz):")
    print("  • These are well below 500 Hz")
    print("  • So they pass through almost unchanged")
    print("  • Maybe 99%+ of original amplitude preserved")
    
    print("\nFor high frequencies (800+ Hz):")
    print("  • These are well above the -3 dB point")
    print("  • Might be attenuated by -40 dB or more")
    print("  • Only 1% or less of original amplitude remains")

def quick_db_calculator():
    """
    Interactive dB calculator
    """
    print("\n" + "="*50)
    print("QUICK dB CALCULATOR")
    print("="*50)
    
    print("\nCommon conversions:")
    test_values = [1.0, 0.707, 0.5, 0.316, 0.1, 0.01, 0.001]
    
    for val in test_values:
        db = 20 * np.log10(val) if val > 0 else float('-inf')
        power_db = 10 * np.log10(val**2) if val > 0 else float('-inf')
        print(f"  Ratio {val:5.3f} = {db:6.1f} dB amplitude = {power_db:6.1f} dB power")
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"   -3 dB amplitude = -6 dB power")
    print(f"   Because power = amplitude²")
    print(f"   So 20×log(0.707) = 10×log(0.707²) = 10×log(0.5) = -3 dB")

if __name__ == "__main__":
    explain_minus_3db()
    quick_db_calculator()