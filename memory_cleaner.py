"""Safe cleanup helpers for project-local cache files.

This module only removes known cache directories under the project root:
`__pycache__` and `.pytest_cache`. It avoids deleting anything outside the
repository and skips system trash cleanup on Windows.
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRS = {".git", ".venv", "venv", "env"}


def _is_within_project(path: Path) -> bool:
    """Return True when `path` is inside the project root."""

    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except (ValueError, OSError):
        return False


def _safe_rmtree(path: Path) -> bool:
    """Remove a directory only if it is a project-local cache folder."""

    if not path.exists() or not path.is_dir():
        return False

    if path.name not in {"__pycache__", ".pytest_cache"}:
        return False

    if not _is_within_project(path):
        logger.warning("Skipped removal outside project root: %s", path)
        return False

    try:
        shutil.rmtree(path)
        logger.debug("Removed: %s", path)
        return True
    except FileNotFoundError:
        return False
    except PermissionError:
        logger.warning("Permission denied: %s", path)
        return False
    except OSError as exc:
        logger.error("Error removing %s: %s", path, exc)
        return False


def clear_pycache_directories():
    """Remove project-local `__pycache__` directories.

    Returns the number of cache directories removed.
    """

    removed_count = 0

    for cache_path in PROJECT_ROOT.rglob("__pycache__"):
        if any(part in EXCLUDED_DIRS for part in cache_path.parts):
            continue
        if _safe_rmtree(cache_path):
            removed_count += 1

    if removed_count > 0:
        logger.info("Memory cleared: %s __pycache__ directories removed.", removed_count)
    else:
        logger.info("No __pycache__ directories found.")

    return removed_count


def clear_pytest_cache():
    """Remove the project-local `.pytest_cache` directory.

    Returns True if the cache was removed, otherwise False.
    """

    cache_path = PROJECT_ROOT / ".pytest_cache"
    removed = _safe_rmtree(cache_path)
    if removed:
        logger.info("Memory cleared: .pytest_cache directory removed.")
    else:
        logger.debug(".pytest_cache not found or skipped.")
    return removed


def clear_trash_files():
    """Skip system trash cleanup on Windows and other non-Linux platforms.

    The project should not try to delete user trash on Windows because the
    trash location is platform-specific and cleanup semantics are risky.
    """

    if platform.system() != "Linux":
        logger.debug("Trash cleanup skipped: not on Linux system.")
        return 0

    trash_path = Path.home() / ".local" / "share" / "Trash" / "files"
    if not trash_path.exists():
        logger.debug("Trash directory not found.")
        return 0

    removed_count = 0
    for item_path in trash_path.iterdir():
        try:
            if item_path.is_dir() and not item_path.is_symlink():
                shutil.rmtree(item_path)
            else:
                item_path.unlink(missing_ok=True)
            removed_count += 1
            logger.debug("Removed from trash: %s", item_path.name)
        except PermissionError:
            logger.warning("Permission denied: %s", item_path)
        except OSError as exc:
            logger.error("Error removing %s: %s", item_path, exc)

    if removed_count > 0:
        logger.info("Memory cleared: %s trash items removed.", removed_count)
    else:
        logger.info("No trash items found.")

    return removed_count


def clear_memory(clean_trash=False):
    """Clear safe project-local caches and optionally Linux trash."""

    logger.info("Starting memory cleanup...")
    logger.info("TEMP START clear_memory clean_trash=%s", clean_trash)

    results = {
        "pycache_removed": clear_pycache_directories(),
        "pytest_cache_removed": clear_pytest_cache(),
        "trash_removed": clear_trash_files() if clean_trash else 0,
    }

    total_items = sum(results.values())
    logger.info("Cleanup complete. Total items removed: %s", total_items)
    logger.info("TEMP END clear_memory results=%s", results)

    return results


def clean():
    """Entry point for running cleanup from the command line."""

    clean_trash = "--trash" in sys.argv or "-t" in sys.argv

    try:
        results = clear_memory(clean_trash=clean_trash)
        print("\nMemory cleanup completed successfully!")
        print(f"  - __pycache__ directories removed: {results['pycache_removed']}")
        print(f"  - .pytest_cache removed: {'Yes' if results['pytest_cache_removed'] else 'No'}")
        if clean_trash:
            print(f"  - Trash items removed: {results['trash_removed']}")
    except Exception as exc:
        logger.error("Fatal error during cleanup: %s", exc)
        print(f"\nMemory cleanup failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    clean()


