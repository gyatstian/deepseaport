import sys
import traceback

from deepseaport.cli import main

if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
        if getattr(sys, "frozen", False):
            try:
                input("\nPress Enter to close...")
            except EOFError:
                pass
    sys.exit(code)
