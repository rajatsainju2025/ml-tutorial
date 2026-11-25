import sys
from pathlib import Path

# nanogpt/ is a standalone project folder (run as `cd nanogpt && python train.py`);
# put it on the path so tests can `import model` / `import train` the same way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nanogpt"))
