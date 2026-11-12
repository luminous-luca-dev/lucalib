"""lucalib – training history recording, plotting, and Discord notifications."""

from .discord_notify import DiscordError, send_discord
from .plot import plot_history
from .recorder import History

__all__ = ["History", "plot_history", "send_discord", "DiscordError"]
