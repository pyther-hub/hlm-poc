import os
import sys

# tests never hit the network; tokenizer must already be in the HF cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
