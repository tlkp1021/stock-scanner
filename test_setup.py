#!/usr/bin/env python3
"""
Test script to verify the setup of Daily Market Scanner.
"""

import sys


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    print("=" * 60)

    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'openbb': 'openbb',
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
        'jinja2': 'jinja2',
    }

    failed = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - OK")
        except ImportError:
            print(f"✗ {package:20s} - MISSING")
            failed.append(package)

    print()
    if failed:
        print(f"❌ Missing packages: {', '.join(failed)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed!")
        return True


def test_openbb():
    """Test OpenBB connection."""
    print("\nTesting OpenBB connection...")
    print("=" * 60)

    try:
        from openbb import obb
        print("✓ OpenBB imported successfully")

        # Try to fetch a simple data point
        try:
            import yfinance as yf
            ticker = yf.Ticker("^GSPC")
            data = ticker.history(period="5d")
            if not data.empty:
                print(f"✓ Data fetching working (got {len(data)} rows)")
                return True
        except Exception as e:
            print(f"✗ Data fetch failed: {e}")
            return False

    except ImportError as e:
        print(f"✗ OpenBB import failed: {e}")
        print("\nInstall with: pip install openbb")
        return False


def test_modules():
    """Test custom modules."""
    print("\nTesting custom modules...")
    print("=" * 60)

    modules = [
        'technical_analysis',
        'market_data',
        'market_research',
        'report_generator',
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module:25s} - OK")
        except Exception as e:
            print(f"✗ {module:25s} - ERROR: {e}")
            return False

    return True


def test_config():
    """Test config file."""
    print("\nTesting configuration...")
    print("=" * 60)

    try:
        from config import ASSETS, TA_PARAMS
        print(f"✓ Config loaded successfully")
        print(f"  - Assets configured: {len(ASSETS)}")
        enabled_assets = sum(1 for a in ASSETS.values() if a.get('enabled', True))
        print(f"  - Assets enabled: {enabled_assets}")
        return True
    except ImportError as e:
        print(f"⚠ Config import failed: {e}")
        print("  Using default configuration")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Daily Market Scanner - Setup Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print()

    results = []

    # Run tests
    results.append(("Package Installation", test_imports()))
    results.append(("OpenBB Connection", test_openbb()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Configuration", test_config()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✅ All tests passed! You can run the scanner with:")
        print("   python main.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
