#!/usr/bin/env python3
"""
Test script to verify SWIFT research environment setup.
Run this to ensure all dependencies are correctly installed.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "accelerate": "Accelerate",
        "datasets": "Datasets",
        "scipy": "SciPy",
        "numpy": "NumPy",
        "pandas": "Pandas",
    }
    
    results = {}
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name:30s} v{version}")
            results[package] = True
        except ImportError as e:
            print(f"✗ {name:30s} FAILED: {e}")
            results[package] = False
    
    return all(results.values())

def test_gpu():
    """Test GPU availability."""
    print("\n" + "=" * 60)
    print("Testing GPU Availability")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA not available (will use CPU - slow!)")
            return False
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_swift_repo():
    """Test that SWIFT repository is accessible."""
    print("\n" + "=" * 60)
    print("Testing SWIFT Repository")
    print("=" * 60)
    
    import os
    
    swift_path = os.path.join(os.path.dirname(__file__), "SWIFT")
    if not os.path.exists(swift_path):
        print(f"✗ SWIFT repository not found at {swift_path}")
        return False
    
    print(f"✓ SWIFT repository found at {swift_path}")
    
    # Check key files
    key_files = [
        "model/swift/modeling_llama.py",
        "model/swift/modeling_mistral.py",
        "evaluation_llama/inference_swift.py",
        "README.md",
    ]
    
    all_exist = True
    for rel_path in key_files:
        full_path = os.path.join(swift_path, rel_path)
        if os.path.exists(full_path):
            print(f"  ✓ {rel_path}")
        else:
            print(f"  ✗ {rel_path} (MISSING)")
            all_exist = False
    
    return all_exist

def test_transformers():
    """Test that we can load a small model."""
    print("\n" + "=" * 60)
    print("Testing Transformers")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        import torch
        
        # Load a tiny model for testing
        print("Loading tiny tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        print("✓ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization works: {len(tokens.input_ids[0])} tokens")
        
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SWIFT Research Environment Test")
    print("=" * 60 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("SWIFT Repository", test_swift_repo),
        ("Transformers", test_transformers),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:10s} - {name}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Environment is ready.")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
