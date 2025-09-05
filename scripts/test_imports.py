#!/usr/bin/env python3
"""
Test script to verify the run_keygraph.py script can be imported
"""

import sys
import os


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

def test_imports():
    """Test that we can import the main script"""
    try:
        import scripts.run_keygraph
        print("✓ Successfully imported run_keygraph.py")
        return True
    except Exception as e:
        print(f"✗ Failed to import run_keygraph.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running KeyGraph script import tests...\n")

    success =test_imports()

    if success:
        print("\n🎉 All import tests passed!")
    else:
        print("\n❌ Some import tests failed!")

    return success

if __name__ =="__main__":
    success =main()
    sys.exit(0 if success else 1)