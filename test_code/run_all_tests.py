#!/usr/bin/env python3
"""
Run surviving test scripts for the TexasPoker framework.
"""

import os
import sys
import subprocess


def run_test(script_name, description):
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Script: {script_name}")
    print('='*60)

    if not os.path.exists(script_name):
        print(f"SKIP  — script not found: {script_name}")
        return None   # not a failure, just absent

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            print(f"PASS  {description}")
            return True
        else:
            print(f"FAIL  {description}")
            return False
    except Exception as e:
        print(f"ERROR {description}: {e}")
        return False


def main():
    tests = [
        ("check_environment.py",   "Environment / dependency check"),
        ("print_npz_content.py",   "Print NPZ file content"),
        ("test_read.py",           "Read test"),
        ("visualize_npz_image.py", "Visualize NPZ image"),
    ]

    results = []
    for script, description in tests:
        outcome = run_test(script, description)
        if outcome is not None:
            results.append((description, outcome))

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for description, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {description}")

    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
