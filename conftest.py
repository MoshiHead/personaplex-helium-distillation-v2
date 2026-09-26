# SPDX-License-Identifier: MIT
"""Makes `import distill` and `import moshi` work when running `pytest` from a
fresh checkout without requiring `pip install -e moshi/` first. If `moshi` is
properly installed (as PersonaPlex_Distill_RunPod.ipynb does), this is a no-op
for it; `distill/` is never pip-installed (it's a sibling of `moshi/`, not a
package inside it), so the path insertion is what makes it importable at all.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

for _path in (_REPO_ROOT, _REPO_ROOT / "moshi"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
