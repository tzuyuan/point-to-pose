import warnings

from .tapir_tracker import TapirTracker
from .cotracker import CoTrackerRealtimeTracker
from .cotracker_offline import CoTrackerOfflineTracker

# Optional trackers: keep the registry importable even if their third-party
# dependencies are not installed.
try:
    from .tapnext_tracker import TapnextTracker
except ImportError as e:
    warnings.warn(f"TAPNext tracker unavailable: {e}")

try:
    from .trackon_tracker import TrackOnTracker
except ImportError as e:
    warnings.warn(f"Track-On tracker unavailable: {e}")

try:
    from .litetracker_tracker import LiteTrackerTracker
except ImportError as e:
    warnings.warn(f"LiteTracker tracker unavailable: {e}")