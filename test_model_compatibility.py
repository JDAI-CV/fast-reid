#!/usr/bin/env python3
"""
Test script to verify model compatibility with different FastReID models.
This script tests both the working veriwild model and themarket_sbs model. 
"""

import torch
import numpy as np
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

def test_model_loading_and_inference(config_path, model_path, model_name):
    """Test model loading and basic inference."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    try:
        #Load configuration.
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = 'cpu'  #Use CPU for testing.
        cfg.freeze()
        
        print("✓ Configuration loaded successfully")
        
        #Build model.
        model = build_model(cfg)
        print("✓ Model built successfully")
        
        #Load weights.
        checkpointer = Checkpointer(model)
        checkpointer.load(model_path)
        print("✓ Model weights loaded successfully")
        
        #Test inference.
        model.eval()
        batch_size = 2
        height, width = cfg.INPUT.SIZE_TEST
        dummy_input = torch.randn(batch_size, 3, height, width)
        
        with torch.no_grad():
            inputs = {"images": dummy_input}
            output = model(inputs)
            
        print(f"✓ Inference successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Feature dimension: {output.shape[1]}")
        
        #Test single image inference.
        single_input = torch.randn(1, 3, height, width)
        with torch.no_grad():
            inputs = {"images": single_input}
            single_output = model(inputs)
            
        print(f"✓ Single image inference successful")
        print(f"  Single input shape: {single_input.shape}")
        print(f"  Single output shape: {single_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("FastReID Model Compatibility Test")
    print("Testing model loading and inference for different model types")
    
    #Test configurations.
    test_configs = [
        {
            "config_path": "veriwild_working_config.yml",
            "model_path": r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
            "model_name": "VeRiWild BOT ResNet50-IBN (Working Model)"
        },
        {
            "config_path": "market_sbs_baseline.yml", 
            "model_path": r"C:\Users\Xeron\Downloads\market_sbs_R101-ibn.pth",
            "model_name": "Market1501 SBS ResNet101-IBN (Previously Problematic)"
        }
    ]
    
    results = []
    for config in test_configs:
        success = test_model_loading_and_inference(
            config["config_path"],
            config["model_path"], 
            config["model_name"]
        )
        results.append((config["model_name"], success))
    
    #Summary.
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for model_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {model_name}")
        if not success:
            all_passed = False
    
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else ' SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n Great. The FastReID library now works with both models!")
        print("The generalizability issues have been resolved.")
    else:
        print("\n Some issues remain. Check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    main()
