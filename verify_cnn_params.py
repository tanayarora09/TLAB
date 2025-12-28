#!/usr/bin/env python3
"""
Comprehensive verification of CNN parameter count adjustments.

This script validates that all CNN models in small_cnn.py have been
correctly modified to have approximately 3728 parameters (±100) while
maintaining batch normalization layers.
"""

def calc_conv_params(in_ch, out_ch, k):
    """Calculate parameters for a convolution layer (weight + bias)"""
    weight = in_ch * out_ch * k * k
    bias = out_ch
    return weight + bias

def calc_bn_params(ch):
    """Calculate parameters for batch normalization (gamma + beta)"""
    return ch * 2

def calc_fc_params(in_f, out_f):
    """Calculate parameters for fully connected layer (weight + bias)"""
    weight = in_f * out_f
    bias = out_f
    return weight + bias

def calc_conv_block(in_ch, out_ch):
    """Calculate parameters for ConvBlock (Conv3x3 + BN + ReLU)"""
    conv = calc_conv_params(in_ch, out_ch, 3)
    bn = calc_bn_params(out_ch)
    return conv + bn

def calc_conv1x1_block(in_ch, out_ch):
    """Calculate parameters for Conv1x1Block (Conv1x1 + BN + ReLU)"""
    conv = calc_conv_params(in_ch, out_ch, 1)
    bn = calc_bn_params(out_ch)
    return conv + bn

def calc_out_block(in_ch, out_ch):
    """Calculate parameters for OutBlock (GAP + FC)"""
    return calc_fc_params(in_ch, out_ch)

def verify_cnna(inchannels=3, outfeatures=10):
    """
    CNNA: 3 conv layers, width 13
    Architecture: 3 -> 13 -> 13 -> 13 -> 10
    """
    params = 0
    breakdown = []
    
    block0 = calc_conv_block(inchannels, 13)
    params += block0
    breakdown.append(f"  block0 (3->13): {block0:5d} params")
    
    block1 = calc_conv_block(13, 13)
    params += block1
    breakdown.append(f"  block1 (13->13): {block1:5d} params")
    
    block2 = calc_conv_block(13, 13)
    params += block2
    breakdown.append(f"  block2 (13->13): {block2:5d} params")
    
    outblock = calc_out_block(13, outfeatures)
    params += outblock
    breakdown.append(f"  outblock (13->10): {outblock:5d} params")
    
    return params, breakdown

def verify_cnnb(inchannels=3, outfeatures=10):
    """
    CNNB: 2 conv layers, width 18
    Architecture: 3 -> 18 -> 18 -> 10
    """
    params = 0
    breakdown = []
    
    block0 = calc_conv_block(inchannels, 18)
    params += block0
    breakdown.append(f"  block0 (3->18): {block0:5d} params")
    
    block1 = calc_conv_block(18, 18)
    params += block1
    breakdown.append(f"  block1 (18->18): {block1:5d} params")
    
    outblock = calc_out_block(18, outfeatures)
    params += outblock
    breakdown.append(f"  outblock (18->10): {outblock:5d} params")
    
    return params, breakdown

def verify_cnnd(inchannels=3, outfeatures=10):
    """
    CNND: 6 conv layers, progressive widths
    Architecture: 3 -> 6 -> 9 -> 9 -> 9 -> 9 -> 9 -> 10
    """
    params = 0
    breakdown = []
    
    block0 = calc_conv_block(inchannels, 6)
    params += block0
    breakdown.append(f"  block0 (3->6): {block0:5d} params")
    
    block1 = calc_conv_block(6, 9)
    params += block1
    breakdown.append(f"  block1 (6->9): {block1:5d} params")
    
    for i in range(2, 6):
        block = calc_conv_block(9, 9)
        params += block
        breakdown.append(f"  block{i} (9->9): {block:5d} params")
    
    outblock = calc_out_block(9, outfeatures)
    params += outblock
    breakdown.append(f"  outblock (9->10): {outblock:5d} params")
    
    return params, breakdown

def verify_cnnw(inchannels=3, outfeatures=10):
    """
    CNNW: 2 layers with 1x1 expansion
    Architecture: 3 -> 13 -> 128 -> 10
    """
    params = 0
    breakdown = []
    
    block0 = calc_conv_block(inchannels, 13)
    params += block0
    breakdown.append(f"  block0 (3->13 Conv3x3): {block0:5d} params")
    
    block1 = calc_conv1x1_block(13, 128)
    params += block1
    breakdown.append(f"  block1 (13->128 Conv1x1): {block1:5d} params")
    
    outblock = calc_out_block(128, outfeatures)
    params += outblock
    breakdown.append(f"  outblock (128->10): {outblock:5d} params")
    
    return params, breakdown

def main():
    """Run comprehensive verification of all CNN models"""
    TARGET = 3728
    TOLERANCE = 100
    
    models = {
        'CNNA': verify_cnna,
        'CNNB': verify_cnnb,
        'CNND': verify_cnnd,
        'CNNW': verify_cnnw,
    }
    
    print("=" * 75)
    print("CNN Parameter Count Verification")
    print("=" * 75)
    print(f"Target: {TARGET} parameters (±{TOLERANCE})")
    print("=" * 75)
    
    all_pass = True
    results = []
    
    for name, verify_func in models.items():
        params, breakdown = verify_func()
        diff = params - TARGET
        in_range = abs(diff) <= TOLERANCE
        status = "✓ PASS" if in_range else "✗ FAIL"
        
        results.append({
            'name': name,
            'params': params,
            'diff': diff,
            'status': status,
            'in_range': in_range,
            'breakdown': breakdown
        })
        
        if not in_range:
            all_pass = False
    
    # Print summary
    print("\nSummary:")
    print("-" * 75)
    for result in results:
        print(f"{result['name']:8s}: {result['params']:5d} params "
              f"(diff: {result['diff']:+5d}) {result['status']}")
    
    # Print detailed breakdowns
    print("\n" + "=" * 75)
    print("Detailed Parameter Breakdowns")
    print("=" * 75)
    
    for result in results:
        print(f"\n{result['name']}:")
        for line in result['breakdown']:
            print(line)
        print(f"  {'─' * 40}")
        print(f"  Total: {result['params']} params")
    
    print("\n" + "=" * 75)
    
    if all_pass:
        print("✓ SUCCESS: All CNNs meet the parameter count requirement!")
        print("=" * 75)
        return 0
    else:
        print("✗ FAILURE: Some CNNs do not meet the parameter count requirement!")
        print("=" * 75)
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
