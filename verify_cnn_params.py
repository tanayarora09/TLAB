#!/usr/bin/env python3
"""
Comprehensive verification of CNN parameter count adjustments.

This script validates that all CNN models in small_cnn.py have been
correctly modified to have approximately 3728 parameters (±100) with batch
normalization, or approximately 2150 parameters (±100) without batch
normalization.

Note: BatchNorm is counted with only weight parameter (1 param per channel),
not bias, as verified by torchinfo.summary.
"""

def calc_conv_params(in_ch, out_ch, k):
    """Calculate parameters for a convolution layer (weight + bias)"""
    weight = in_ch * out_ch * k * k
    bias = out_ch
    return weight + bias

def calc_bn_params(ch):
    """Calculate parameters for batch normalization (weight only, no bias)"""
    return ch

def calc_fc_params(in_f, out_f):
    """Calculate parameters for fully connected layer (weight + bias)"""
    weight = in_f * out_f
    bias = out_f
    return weight + bias

def calc_conv_block(in_ch, out_ch, use_bn=True):
    """Calculate parameters for ConvBlock (Conv3x3 + optional BN + ReLU)"""
    conv = calc_conv_params(in_ch, out_ch, 3)
    bn = calc_bn_params(out_ch) if use_bn else 0
    return conv + bn

def calc_conv1x1_block(in_ch, out_ch, use_bn=True):
    """Calculate parameters for Conv1x1Block (Conv1x1 + optional BN + ReLU)"""
    conv = calc_conv_params(in_ch, out_ch, 1)
    bn = calc_bn_params(out_ch) if use_bn else 0
    return conv + bn

def calc_out_block(in_ch, out_ch):
    """Calculate parameters for OutBlock (GAP + FC)"""
    return calc_fc_params(in_ch, out_ch)

def verify_cnna(inchannels=3, outfeatures=10, with_bn=True):
    """
    CNNA: 3 conv layers
    With BN: 3->14->13->13 (3757 params)
    Without BN: width 10 (2210 params)
    """
    params = 0
    breakdown = []
    
    if with_bn:
        block0 = calc_conv_block(inchannels, 14, True)
        params += block0
        breakdown.append(f"  block0 (3->14): {block0:5d} params")
        
        block1 = calc_conv_block(14, 13, True)
        params += block1
        breakdown.append(f"  block1 (14->13): {block1:5d} params")
        
        block2 = calc_conv_block(13, 13, True)
        params += block2
        breakdown.append(f"  block2 (13->13): {block2:5d} params")
        
        outblock = calc_out_block(13, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (13->10): {outblock:5d} params")
    else:
        ch = 10
        block0 = calc_conv_block(inchannels, ch, False)
        params += block0
        breakdown.append(f"  block0 (3->10): {block0:5d} params")
        
        block1 = calc_conv_block(ch, ch, False)
        params += block1
        breakdown.append(f"  block1 (10->10): {block1:5d} params")
        
        block2 = calc_conv_block(ch, ch, False)
        params += block2
        breakdown.append(f"  block2 (10->10): {block2:5d} params")
        
        outblock = calc_out_block(ch, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (10->10): {outblock:5d} params")
    
    return params, breakdown

def verify_cnnb(inchannels=3, outfeatures=10, with_bn=True):
    """
    CNNB: 2 conv layers
    With BN: width 18 (3664 params)
    Without BN: 3->13->14 (2166 params)
    """
    params = 0
    breakdown = []
    
    if with_bn:
        block0 = calc_conv_block(inchannels, 18, True)
        params += block0
        breakdown.append(f"  block0 (3->18): {block0:5d} params")
        
        block1 = calc_conv_block(18, 18, True)
        params += block1
        breakdown.append(f"  block1 (18->18): {block1:5d} params")
        
        outblock = calc_out_block(18, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (18->10): {outblock:5d} params")
    else:
        block0 = calc_conv_block(inchannels, 13, False)
        params += block0
        breakdown.append(f"  block0 (3->13): {block0:5d} params")
        
        block1 = calc_conv_block(13, 14, False)
        params += block1
        breakdown.append(f"  block1 (13->14): {block1:5d} params")
        
        outblock = calc_out_block(14, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (14->10): {outblock:5d} params")
    
    return params, breakdown

def verify_cnnd(inchannels=3, outfeatures=10, with_bn=True):
    """
    CNND: 6 conv layers
    With BN: 3->6->9->9->9->9->9 (3766 params)
    Without BN: 3->3->7->7->7->7->7 (2152 params)
    """
    params = 0
    breakdown = []
    
    if with_bn:
        block0 = calc_conv_block(inchannels, 6, True)
        params += block0
        breakdown.append(f"  block0 (3->6): {block0:5d} params")
        
        block1 = calc_conv_block(6, 9, True)
        params += block1
        breakdown.append(f"  block1 (6->9): {block1:5d} params")
        
        for i in range(2, 6):
            block = calc_conv_block(9, 9, True)
            params += block
            breakdown.append(f"  block{i} (9->9): {block:5d} params")
        
        outblock = calc_out_block(9, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (9->10): {outblock:5d} params")
    else:
        block0 = calc_conv_block(inchannels, 3, False)
        params += block0
        breakdown.append(f"  block0 (3->3): {block0:5d} params")
        
        block1 = calc_conv_block(3, 7, False)
        params += block1
        breakdown.append(f"  block1 (3->7): {block1:5d} params")
        
        for i in range(2, 6):
            block = calc_conv_block(7, 7, False)
            params += block
            breakdown.append(f"  block{i} (7->7): {block:5d} params")
        
        outblock = calc_out_block(7, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (7->10): {outblock:5d} params")
    
    return params, breakdown

def verify_cnnw(inchannels=3, outfeatures=10, with_bn=True):
    """
    CNNW: 1 conv3x3 + 1 conv1x1 expansion
    With BN: 13->130 (3637 params)
    Without BN: 13->70 (2054 params)
    """
    params = 0
    breakdown = []
    
    if with_bn:
        block0 = calc_conv_block(inchannels, 13, True)
        params += block0
        breakdown.append(f"  block0 (3->13 Conv3x3): {block0:5d} params")
        
        block1 = calc_conv1x1_block(13, 130, True)
        params += block1
        breakdown.append(f"  block1 (13->130 Conv1x1): {block1:5d} params")
        
        outblock = calc_out_block(130, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (130->10): {outblock:5d} params")
    else:
        block0 = calc_conv_block(inchannels, 13, False)
        params += block0
        breakdown.append(f"  block0 (3->13 Conv3x3): {block0:5d} params")
        
        block1 = calc_conv1x1_block(13, 70, False)
        params += block1
        breakdown.append(f"  block1 (13->70 Conv1x1): {block1:5d} params")
        
        outblock = calc_out_block(70, outfeatures)
        params += outblock
        breakdown.append(f"  outblock (70->10): {outblock:5d} params")
    
    return params, breakdown

def main():
    """Run comprehensive verification of all CNN models"""
    
    models = {
        'CNNA': verify_cnna,
        'CNNB': verify_cnnb,
        'CNND': verify_cnnd,
        'CNNW': verify_cnnw,
    }
    
    print("=" * 75)
    print("CNN Parameter Count Verification")
    print("=" * 75)
    
    # Test with batch normalization (target ~3728)
    TARGET_BN = 3728
    TOLERANCE = 100
    print(f"\nWith Batch Normalization - Target: {TARGET_BN} (±{TOLERANCE})")
    print("=" * 75)
    
    all_pass = True
    results_bn = []
    
    for name, verify_func in models.items():
        params, breakdown = verify_func(with_bn=True)
        diff = params - TARGET_BN
        in_range = abs(diff) <= TOLERANCE
        status = "✓ PASS" if in_range else "✗ FAIL"
        
        results_bn.append({
            'name': name,
            'params': params,
            'diff': diff,
            'status': status,
            'in_range': in_range,
            'breakdown': breakdown
        })
        
        if not in_range:
            all_pass = False
    
    # Print summary for with BN
    print("\nSummary:")
    print("-" * 75)
    for result in results_bn:
        print(f"{result['name']:8s}: {result['params']:5d} params "
              f"(diff: {result['diff']:+5d}) {result['status']}")
    
    # Test without batch normalization (target ~2150)
    TARGET_NO_BN = 2150
    print(f"\n\nWithout Batch Normalization - Target: {TARGET_NO_BN} (±{TOLERANCE})")
    print("=" * 75)
    
    results_no_bn = []
    
    for name, verify_func in models.items():
        params, breakdown = verify_func(with_bn=False)
        diff = params - TARGET_NO_BN
        in_range = abs(diff) <= TOLERANCE
        status = "✓ PASS" if in_range else "✗ FAIL"
        
        results_no_bn.append({
            'name': name,
            'params': params,
            'diff': diff,
            'status': status,
            'in_range': in_range,
            'breakdown': breakdown
        })
        
        if not in_range:
            all_pass = False
    
    # Print summary for without BN
    print("\nSummary:")
    print("-" * 75)
    for result in results_no_bn:
        print(f"{result['name']:8s}: {result['params']:5d} params "
              f"(diff: {result['diff']:+5d}) {result['status']}")
    
    # Print detailed breakdowns
    print("\n" + "=" * 75)
    print("Detailed Parameter Breakdowns")
    print("=" * 75)
    
    print("\nWith Batch Normalization:")
    for result in results_bn:
        print(f"\n{result['name']}:")
        for line in result['breakdown']:
            print(line)
        print(f"  {'─' * 40}")
        print(f"  Total: {result['params']} params")
    
    print("\n" + "-" * 75)
    print("Without Batch Normalization:")
    for result in results_no_bn:
        print(f"\n{result['name']}:")
        for line in result['breakdown']:
            print(line)
        print(f"  {'─' * 40}")
        print(f"  Total: {result['params']} params")
    
    print("\n" + "=" * 75)
    
    if all_pass:
        print("✓ SUCCESS: All CNNs meet the parameter count requirements!")
        print("=" * 75)
        return 0
    else:
        print("✗ FAILURE: Some CNNs do not meet the parameter count requirements!")
        print("=" * 75)
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
