#!/usr/bin/env python3
"""
Scheduler for Daily Market Scanner
Sets up automatic daily execution.
"""

import sys
import platform


def setup_windows_scheduler():
    """Create a Windows Task Scheduler task."""
    import os

    python_path = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'main.py')

    print("Windows Task Scheduler Setup")
    print("=" * 60)
    print("\nTo set up a daily task, follow these steps:")
    print("\n1. Open Task Scheduler (Windows Key + R, type 'taskschd.msc')")
    print("2. Click 'Create Basic Task' on the right")
    print("3. Name: 'Daily Market Scanner'")
    print("4. Trigger: Daily")
    print("5. Start time: Your preferred time (e.g., 8:00 AM)")
    print("6. Action: 'Start a program'")
    print(f"7. Program/script: {python_path}")
    print(f"8. Add arguments: {main_script}")
    print(f"9. Start in: {script_dir}")
    print("\nAlternatively, you can create an XML task import file.")

    # Generate batch file for manual run
    batch_file = os.path.join(script_dir, 'run_scanner.bat')
    with open(batch_file, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'cd /d "{script_dir}"\n')
        f.write(f'"{python_path}" main.py\n')
        f.write(f'pause\n')

    print(f"\n✅ Created batch file: {batch_file}")
    print("   You can double-click this file to run the scanner manually.")


def setup_linux_cron():
    """Create a cron job for Linux/Mac."""
    import os

    python_path = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'main.py')

    print("Linux/Mac Cron Setup")
    print("=" * 60)

    # Generate shell script
    shell_script = os.path.join(script_dir, 'run_scanner.sh')
    with open(shell_script, 'w') as f:
        f.write(f'#!/bin/bash\n')
        f.write(f'cd "{script_dir}"\n')
        f.write(f'"{python_path}" main.py\n')
    os.chmod(shell_script, 0o755)

    print(f"\n✅ Created shell script: {shell_script}")
    print("\nTo add a cron job, run:")
    print("  crontab -e")
    print("\nAnd add this line (runs daily at 8:00 AM):")
    print(f"  0 8 * * * {shell_script}")


def main():
    """Setup scheduler based on platform."""
    system = platform.system()

    print("\n" + "=" * 60)
    print("Daily Market Scanner - Scheduler Setup")
    print("=" * 60)
    print(f"\nDetected OS: {system}")
    print(f"Python path: {sys.executable}")

    if system == 'Windows':
        setup_windows_scheduler()
    else:
        setup_linux_cron()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
