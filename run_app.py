#!/usr/bin/env python3
"""
Streamlit Web App Launcher for Cross-Market Prediction
"""
import os
import subprocess
import sys


def run_streamlit_app():
    """Launch the Streamlit web application"""
    
    # Get the current directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(app_dir, "streamlit_app.py")
    
    print("=" * 70)
    print("🚀 Starting Cross-Market Prediction Web Application")
    print("=" * 70)
    print()
    print("📍 App Location:", app_file)
    print()
    print("⏳ Launching Streamlit server...")
    print()
    print("💡 Once the app loads, you can access it at:")
    print("   👉 http://localhost:8501")
    print()
    print("ℹ️  Press Ctrl+C to stop the server")
    print()
    print("=" * 70)
    print()
    
    try:
        # Run streamlit app
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_file, "--logger.level=error"],
            cwd=app_dir,
            check=False
        )
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
