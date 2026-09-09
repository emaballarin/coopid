"""Controller-shaped Lagrange multipliers for Cooper.

Additive only: every object here is a Cooper object or produces one, and stock Cooper keeps
working unchanged. See `PROJECT.md` for scope and build order.
"""

from importlib.metadata import PackageNotFoundError, version

from coopid.filters import EMAViolation
from coopid.multipliers import BoundedMultiplier
from coopid.schedules import GatedLevel, OpenLoopLevel

__all__ = ["BoundedMultiplier", "EMAViolation", "GatedLevel", "OpenLoopLevel"]

try:
    __version__ = version("coopid")
except PackageNotFoundError:  # not installed; e.g. running from a source checkout
    __version__ = "0.0.0.dev0"

del version, PackageNotFoundError
