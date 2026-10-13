"""命令行入口：python -m imgcrop 等价于 python -m imgcrop.cli。"""

from __future__ import annotations

import sys

from imgcrop.cli import main

if __name__ == "__main__":
    sys.exit(main())
