"""
HuggingFace Download Manager
=============================
A production-quality desktop application for downloading Hugging Face models,
especially large GGUF files, with resume support, aria2c integration, and
a beginner-friendly modern UI.

Architecture:
  - DownloadManager   : Core download logic (aria2c + hf_hub)
  - SettingsManager   : Persistent settings (JSON)
  - TokenManager      : Secure HF token storage
  - QueueManager      : Download queue and state
  - MainWindow        : PySide6/PyQt6 main UI
  - Each tab is its own widget class for modularity

Author: HF Download Manager
License: MIT
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import os
import json
import re
import time
import logging
import threading
import subprocess
import platform
import shutil
import hashlib
import urllib.request
import urllib.parse
import zipfile
import textwrap
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# ── Qt import (try PySide6 first, fall back to PyQt6) ────────────────────────
QT_BINDING = None
QT_ERROR_MSG = ""

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QLineEdit, QTextEdit, QProgressBar,
        QTabWidget, QScrollArea, QFrame, QSplitter, QListWidget,
        QListWidgetItem, QComboBox, QCheckBox, QSpinBox, QSlider,
        QFileDialog, QMessageBox, QSystemTrayIcon, QMenu,            # ← QAction removed
        QToolTip, QSizePolicy, QGroupBox, QGridLayout, QStackedWidget,
        QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QAbstractItemView, QStatusBar, QToolBar,
        QTreeWidget, QTreeWidgetItem, QRadioButton, QButtonGroup,
        QDoubleSpinBox
    )
    from PySide6.QtCore import (
        Qt, QThread, Signal, QTimer, QSize, QUrl, QSettings,
        QPropertyAnimation, QEasingCurve, QPoint, QRect, QMimeData,
        QSortFilterProxyModel, QAbstractTableModel, QModelIndex
    )
    from PySide6.QtGui import (
        QIcon, QPixmap, QColor, QFont, QPalette, QCursor, QClipboard,
        QDesktopServices, QFontDatabase, QPainter, QLinearGradient,
        QBrush, QPen, QTextCursor, QAction, QAction as QGuiAction   # ← QAction added
    )
    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QLineEdit, QTextEdit, QProgressBar,
            QTabWidget, QScrollArea, QFrame, QSplitter, QListWidget,
            QListWidgetItem, QComboBox, QCheckBox, QSpinBox, QSlider,
            QFileDialog, QMessageBox, QSystemTrayIcon, QMenu,            # ← QAction removed
            QToolTip, QSizePolicy, QGroupBox, QGridLayout, QStackedWidget,
            QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem,
            QHeaderView, QAbstractItemView, QStatusBar, QToolBar,
            QTreeWidget, QTreeWidgetItem, QRadioButton, QButtonGroup,
            QDoubleSpinBox
        )
        from PyQt6.QtCore import (
            Qt, QThread, pyqtSignal as Signal, QTimer, QSize, QUrl, QSettings,
            QPropertyAnimation, QEasingCurve, QPoint, QRect, QMimeData,
            QSortFilterProxyModel, QAbstractTableModel, QModelIndex
        )
        from PyQt6.QtGui import (
            QIcon, QPixmap, QColor, QFont, QPalette, QCursor, QClipboard,
            QDesktopServices, QFontDatabase, QPainter, QLinearGradient,
            QBrush, QPen, QTextCursor, QAction, QAction as QGuiAction   # ← QAction added
        )
        QT_BINDING = "PyQt6"
    except ImportError:
        QT_BINDING = None
        QT_ERROR_MSG = (
            "\n❌ No Qt binding found!\n"
            "Please install PySide6 or PyQt6:\n"
            "   pip install PySide6\n"
            "or\n"
            "   pip install PyQt6\n"
        )

# ── Early exit if Qt is unavailable ─────────────────────────────────────────
if QT_BINDING is None:
    print(QT_ERROR_MSG)
    sys.exit(1)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path.home() / ".hf_download_manager" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"hfdm_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("HFDM")

# ── App Constants ─────────────────────────────────────────────────────────────
APP_NAME    = "HuggingFace Download Manager"
APP_VERSION = "1.0.0"
APP_DIR     = Path.home() / ".hf_download_manager"
SETTINGS_FILE = APP_DIR / "settings.json"
QUEUE_FILE    = APP_DIR / "queue.json"
TOKEN_FILE    = APP_DIR / "token.enc"
ARIA2_DIR     = APP_DIR / "aria2"

# ── Enums ─────────────────────────────────────────────────────────────────────
class DownloadStatus(Enum):
    QUEUED    = "Queued"
    ACTIVE    = "Downloading"
    PAUSED    = "Paused"
    COMPLETE  = "Complete"
    FAILED    = "Failed"
    VERIFYING = "Verifying"
    RETRYING  = "Retrying"

class DownloadMethod(Enum):
    AUTO     = "Auto (Recommended)"
    HF_HUB   = "HuggingFace Hub"
    ARIA2    = "aria2c (Direct)"

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class DownloadTask:
    """Represents a single download task in the queue."""
    id:          str
    url:         str
    filename:    str
    dest_dir:    str
    method:      str          = DownloadMethod.AUTO.value
    status:      str          = DownloadStatus.QUEUED.value
    progress:    float        = 0.0        # 0-100
    speed:       float        = 0.0        # bytes/sec
    eta:         int          = 0          # seconds
    size:        int          = 0          # bytes total
    downloaded:  int          = 0          # bytes done
    retries:     int          = 0
    max_retries: int          = 5
    error:       str          = ""
    created_at:  str          = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str          = ""
    connections: int          = 16
    repo_id:     str          = ""
    repo_type:   str          = "model"
    hf_filename: str          = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DownloadTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class SettingsManager:
    """Manages persistent application settings stored as JSON."""

    DEFAULTS: Dict[str, Any] = {
        "download_dir":       str(Path.home() / "Downloads" / "HF_Models"),
        "temp_dir":           str(APP_DIR / "temp"),
        "connections":        16,
        "max_retries":        10,
        "retry_wait":         5,
        "speed_limit":        0,          # 0 = unlimited (bytes/sec)
        "parallel_downloads": 2,
        "chunk_size_mb":      8,
        "auto_retry":         True,
        "verify_hash":        True,
        "dark_mode":          True,
        "simple_mode":        True,
        "tray_minimize":      True,
        "clipboard_detect":   True,
        "auto_shutdown":      False,
        "proxy":              "",
        "aria2_extra_args":   "",
        "hf_endpoint":        "https://huggingface.co",
        "theme":              "dark",
        "notifications":      True,
        "show_advanced":      False,
    }

    def __init__(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = dict(self.DEFAULTS)
        self.load()

    def load(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data.update(saved)
                log.debug("Settings loaded.")
            except Exception as e:
                log.warning(f"Could not load settings: {e}")

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            log.error(f"Could not save settings: {e}")

    def get(self, key: str, default=None):
        return self._data.get(key, self.DEFAULTS.get(key, default))

    def set(self, key: str, value: Any):
        self._data[key] = value
        self.save()

    def reset(self):
        self._data = dict(self.DEFAULTS)
        self.save()


# ══════════════════════════════════════════════════════════════════════════════
#  TOKEN MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class TokenManager:
    """Manages HuggingFace API token storage (simple obfuscation, not encryption)."""

    def __init__(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)

    def save_token(self, token: str):
        """Save token with simple XOR obfuscation."""
        try:
            key = b"HFDM_SECRET_KEY_2024"
            encoded = bytearray()
            for i, c in enumerate(token.encode()):
                encoded.append(c ^ key[i % len(key)])
            with open(TOKEN_FILE, "wb") as f:
                f.write(encoded)
            log.info("Token saved.")
        except Exception as e:
            log.error(f"Could not save token: {e}")

    def load_token(self) -> Optional[str]:
        """Load and decode saved token."""
        if not TOKEN_FILE.exists():
            return None
        try:
            key = b"HFDM_SECRET_KEY_2024"
            with open(TOKEN_FILE, "rb") as f:
                encoded = f.read()
            decoded = bytearray()
            for i, c in enumerate(encoded):
                decoded.append(c ^ key[i % len(key)])
            return decoded.decode()
        except Exception as e:
            log.error(f"Could not load token: {e}")
            return None

    def delete_token(self):
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
            log.info("Token deleted.")

    def validate_token(self, token: str) -> bool:
        """Quick format validation (starts with hf_)."""
        return token.startswith("hf_") and len(token) > 10


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class QueueManager:
    """Manages the download queue with persistence."""

    def __init__(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self.tasks: List[DownloadTask] = []
        self.load()

    def load(self):
        if QUEUE_FILE.exists():
            try:
                with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tasks = [DownloadTask.from_dict(d) for d in data]
                # Reset active downloads to queued on restart
                for t in self.tasks:
                    if t.status == DownloadStatus.ACTIVE.value:
                        t.status = DownloadStatus.QUEUED.value
                log.info(f"Queue loaded: {len(self.tasks)} tasks.")
            except Exception as e:
                log.warning(f"Could not load queue: {e}")

    def save(self):
        try:
            with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump([t.to_dict() for t in self.tasks], f, indent=2)
        except Exception as e:
            log.error(f"Could not save queue: {e}")

    def add(self, task: DownloadTask):
        self.tasks.append(task)
        self.save()

    def remove(self, task_id: str):
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.save()

    def get(self, task_id: str) -> Optional[DownloadTask]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def next_queued(self) -> Optional[DownloadTask]:
        return next((t for t in self.tasks if t.status == DownloadStatus.QUEUED.value), None)

    def active_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == DownloadStatus.ACTIVE.value)

    def update(self, task: DownloadTask):
        for i, t in enumerate(self.tasks):
            if t.id == task.id:
                self.tasks[i] = task
                break
        self.save()


# ══════════════════════════════════════════════════════════════════════════════
#  ARIA2 MANAGER  (detect / download / run)
# ══════════════════════════════════════════════════════════════════════════════
class Aria2Manager:
    """Handles aria2c binary detection and download."""

    ARIA2_WINDOWS_URL = (
        "https://github.com/aria2/aria2/releases/download/release-1.37.0/"
        "aria2-1.37.0-win-64bit-build1.zip"
    )
    ARIA2_BINARY = "aria2c.exe" if platform.system() == "Windows" else "aria2c"

    def __init__(self):
        self.binary_path: Optional[str] = None
        self._detect()

    def _detect(self):
        # Check PATH first
        found = shutil.which("aria2c")
        if found:
            self.binary_path = found
            log.info(f"aria2c found in PATH: {found}")
            return
        # Check bundled location
        bundled = ARIA2_DIR / self.ARIA2_BINARY
        if bundled.exists():
            self.binary_path = str(bundled)
            log.info(f"aria2c found bundled: {bundled}")
            return
        log.warning("aria2c not found.")

    @property
    def available(self) -> bool:
        return self.binary_path is not None

    def download_aria2(self, progress_callback=None) -> bool:
        """Download and extract aria2c for Windows."""
        if platform.system() != "Windows":
            log.error("Auto-download only supported on Windows.")
            return False
        try:
            ARIA2_DIR.mkdir(parents=True, exist_ok=True)
            zip_path = ARIA2_DIR / "aria2.zip"
            log.info(f"Downloading aria2c from {self.ARIA2_WINDOWS_URL}")

            def _report(count, block, total):
                if total > 0 and progress_callback:
                    pct = min(100, int(count * block * 100 / total))
                    progress_callback(pct, f"Downloading aria2c... {pct}%")

            urllib.request.urlretrieve(self.ARIA2_WINDOWS_URL, zip_path, _report)

            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.namelist():
                    if member.endswith("aria2c.exe"):
                        z.extract(member, ARIA2_DIR)
                        extracted = ARIA2_DIR / member
                        final = ARIA2_DIR / "aria2c.exe"
                        shutil.move(str(extracted), str(final))
                        break

            zip_path.unlink(missing_ok=True)
            self._detect()
            return self.available
        except Exception as e:
            log.error(f"Failed to download aria2c: {e}")
            return False

    def get_version(self) -> str:
        if not self.available:
            return "Not installed"
        try:
            result = subprocess.run(
                [self.binary_path, "--version"],
                capture_output=True, text=True, timeout=5
            )
            line = result.stdout.splitlines()[0] if result.stdout else ""
            return line.strip() or "Unknown"
        except Exception:
            return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY CHECKER
# ══════════════════════════════════════════════════════════════════════════════
class DependencyChecker:
    """Checks and installs missing Python dependencies."""

    REQUIRED = ["requests", "huggingface_hub", "tqdm"]

    @staticmethod
    def check_all() -> Dict[str, bool]:
        results = {}
        for pkg in DependencyChecker.REQUIRED:
            try:
                __import__(pkg.replace("-", "_"))
                results[pkg] = True
            except ImportError:
                results[pkg] = False
        return results

    @staticmethod
    def install(pkg: str, progress_callback=None) -> bool:
        try:
            if progress_callback:
                progress_callback(0, f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                timeout=120
            )
            if progress_callback:
                progress_callback(100, f"{pkg} installed!")
            return True
        except Exception as e:
            log.error(f"Failed to install {pkg}: {e}")
            return False

    @staticmethod
    def install_all_missing(progress_callback=None) -> bool:
        status = DependencyChecker.check_all()
        missing = [k for k, v in status.items() if not v]
        if not missing:
            return True
        for pkg in missing:
            if not DependencyChecker.install(pkg, progress_callback):
                return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  HF API HELPER
# ══════════════════════════════════════════════════════════════════════════════
class HFApiHelper:
    """Wraps huggingface_hub calls safely."""

    @staticmethod
    def list_repo_files(repo_id: str, token: Optional[str] = None) -> List[Dict]:
        """Returns list of {name, size, lfs} dicts for a repo."""
        try:
            from huggingface_hub import list_repo_tree, hf_hub_url, get_hf_file_metadata
            results = []
            for item in list_repo_tree(repo_id, token=token, recursive=True):
                if hasattr(item, 'path') and hasattr(item, 'size'):
                    results.append({
                        "name": item.path,
                        "size": item.size or 0,
                        "is_lfs": getattr(item, 'lfs', None) is not None,
                    })
            return results
        except Exception as e:
            log.error(f"list_repo_files error: {e}")
            return []

    @staticmethod
    def get_file_url(repo_id: str, filename: str, token: Optional[str] = None) -> str:
        try:
            from huggingface_hub import hf_hub_url
            return hf_hub_url(repo_id, filename, token=token)
        except Exception as e:
            log.error(f"get_file_url error: {e}")
            return ""

    @staticmethod
    def validate_token(token: str) -> bool:
        try:
            from huggingface_hub import whoami
            info = whoami(token=token)
            return bool(info.get("name"))
        except Exception:
            return False

    @staticmethod
    def get_user_info(token: str) -> Optional[Dict]:
        try:
            from huggingface_hub import whoami
            return whoami(token=token)
        except Exception:
            return None

    @staticmethod
    def search_models(query: str, filter_gguf: bool = True, limit: int = 30,
                      token: Optional[str] = None) -> List[Dict]:
        try:
            from huggingface_hub import list_models
            tags = ["gguf"] if filter_gguf else []
            models = list(list_models(
                search=query,
                tags=tags,
                limit=limit,
                token=token,
                sort="downloads",
                direction=-1,
            ))
            return [
                {
                    "id": m.id,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "tags": getattr(m, "tags", []),
                    "last_modified": str(getattr(m, "last_modified", "")),
                }
                for m in models
            ]
        except Exception as e:
            log.error(f"search_models error: {e}")
            return []


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD WORKER  (QThread)
# ══════════════════════════════════════════════════════════════════════════════
class DownloadWorker(QThread):
    """
    Background thread that performs the actual download.
    Emits signals to update the UI.
    """
    progress_signal  = Signal(str, float, float, float, float, float)   # id, %, speed, eta, downloaded, total (floats avoid overflow)
    status_signal    = Signal(str, str)                            # id, status string
    log_signal       = Signal(str, str)                            # id, message
    finished_signal  = Signal(str, bool, str)                     # id, success, error

    def __init__(self, task: DownloadTask, settings: SettingsManager,
                 aria2: Aria2Manager, token: Optional[str] = None):
        super().__init__()
        self.task     = task
        self.settings = settings
        self.aria2    = aria2
        self.token    = token
        self._stop    = threading.Event()
        self._pause   = threading.Event()

    def stop(self):
        self._stop.set()

    def pause(self):
        self._pause.set()

    def resume_dl(self):
        self._pause.clear()

    def run(self):
        task = self.task
        self.status_signal.emit(task.id, DownloadStatus.ACTIVE.value)
        self.log_signal.emit(task.id, f"Starting download: {task.filename}")

        dest = Path(task.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # Determine method
        method = task.method
        if method == DownloadMethod.AUTO.value:
            # Use hf_hub if it looks like a HF repo, else aria2
            if task.repo_id:
                method = DownloadMethod.HF_HUB.value
            elif self.aria2.available:
                method = DownloadMethod.ARIA2.value
            else:
                method = DownloadMethod.HF_HUB.value

        success = False
        error = ""
        for attempt in range(1, task.max_retries + 1):
            if self._stop.is_set():
                break
            if attempt > 1:
                self.status_signal.emit(task.id, DownloadStatus.RETRYING.value)
                self.log_signal.emit(task.id, f"Retry {attempt}/{task.max_retries}...")
                time.sleep(self.settings.get("retry_wait", 5))

            try:
                if method == DownloadMethod.ARIA2.value and self.aria2.available:
                    success, error = self._download_aria2(task, dest)
                else:
                    success, error = self._download_hf_hub(task, dest)
                if success:
                    break
            except Exception as e:
                error = str(e)
                log.error(f"Download attempt {attempt} failed: {e}")

        if success:
            self.status_signal.emit(task.id, DownloadStatus.COMPLETE.value)
            self.log_signal.emit(task.id, "✅ Download complete!")
        else:
            self.status_signal.emit(task.id, DownloadStatus.FAILED.value)
            self.log_signal.emit(task.id, f"❌ Failed: {error}")

        self.finished_signal.emit(task.id, success, error)

    # ── aria2c download ───────────────────────────────────────────────────────
    def _download_aria2(self, task: DownloadTask, dest: Path):
        """Download using aria2c subprocess with progress parsing."""
        cmd = [
            self.aria2.binary_path,
            "--continue=true",
            f"--max-connection-per-server={task.connections}",
            f"--split={task.connections}",
            f"--min-split-size=5M",
            f"--dir={dest}",
            f"--out={task.filename}",
            f"--max-tries={task.max_retries}",
            "--retry-wait=5",
            "--connect-timeout=30",
            "--timeout=30",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--check-integrity=true",
            "--console-log-level=notice",
            "--summary-interval=1",
        ]

        # Speed limit
        limit = self.settings.get("speed_limit", 0)
        if limit > 0:
            cmd.append(f"--max-download-limit={limit}")

        # Token auth header for HF
        if self.token and "huggingface.co" in task.url:
            cmd.append(f"--header=Authorization: Bearer {self.token}")

        # Extra user args
        extra = self.settings.get("aria2_extra_args", "").strip()
        if extra:
            cmd.extend(extra.split())

        # Proxy
        proxy = self.settings.get("proxy", "").strip()
        if proxy:
            cmd.append(f"--all-proxy={proxy}")

        cmd.append(task.url)

        self.log_signal.emit(task.id, f"Running aria2c with {task.connections} connections...")
        log.debug(f"aria2 cmd: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            last_progress = 0.0
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    return False, "Cancelled by user"

                while self._pause.is_set():
                    time.sleep(0.5)

                line = line.strip()
                # Parse aria2 progress line: [#xxxx 100MiB/1GiB(10%) CN:16 DL:5MiB ETA:3m]
                m = re.search(
                    r'\[#\w+\s+([\d.]+\w+)/([\d.]+\w+)\((\d+)%\)\s+CN:(\d+)\s+DL:([\d.]+\w+)\s+ETA:(\S+)\]',
                    line
                )
                if m:
                    downloaded_str = m.group(1)
                    total_str      = m.group(2)
                    pct            = float(m.group(3))
                    cn             = int(m.group(4))
                    speed_str      = m.group(5)
                    eta_str        = m.group(6)

                    speed_bytes = self._parse_size(speed_str + "/s") if speed_str else 0
                    eta_sec     = self._parse_eta(eta_str)
                    dl_bytes    = self._parse_size(downloaded_str)
                    total_bytes = self._parse_size(total_str)

                    self.progress_signal.emit(
                        task.id, pct, speed_bytes, eta_sec, dl_bytes, total_bytes
                    )
                    last_progress = pct

                if line:
                    self.log_signal.emit(task.id, line)

            proc.wait()
            return proc.returncode == 0, f"aria2c exit code {proc.returncode}"

        except FileNotFoundError:
            return False, "aria2c binary not found"
        except Exception as e:
            return False, str(e)

    # ── huggingface_hub download ──────────────────────────────────────────────
    # ── huggingface_hub download (reliable, resumable, token‑aware) ──────────
    def _download_hf_hub(self, task: DownloadTask, dest: Path):
        """Download using huggingface_hub official downloader – handles resume, auth, and URL renewal."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            return False, f"Missing huggingface_hub: {e}"

        try:
            # Use the built-in downloader that manages everything
            output_path = hf_hub_download(
                repo_id=task.repo_id,
                filename=task.hf_filename,
                subfolder=None,
                local_dir=dest,
                local_dir_use_symlinks=False,
                token=self.token,
                resume=True,
                progress=True,
                # Force download to final destination instead of cache
                local_files_only=False,
                # We need a progress callback to feed into our UI
                progress_callback=self._hf_progress_callback(task)
            )
            # Success – the file is already at output_path
            self.log_signal.emit(task.id, f"Downloaded to {output_path}")
            return True, ""
        except Exception as e:
            return False, str(e)

    def _hf_progress_callback(self, task: DownloadTask):
        """Return a progress function compatible with hf_hub_download."""
        start_time = time.time()
        last_update = start_time
        def callback(current: int, total: int):
            nonlocal last_update
            now = time.time()
            if now - last_update < 0.5 and current < total:
                return
            last_update = now
            # Estimate speed and ETA
            elapsed = max(now - start_time, 0.001)
            speed = current / elapsed
            remaining = (total - current) / speed if speed > 0 else 0
            pct = (current / total * 100) if total > 0 else 0
            self.progress_signal.emit(
                task.id, pct, speed, int(remaining), current, total
            )
            self.log_signal.emit(task.id,
                f"Downloading... {current}/{total} ({pct:.1f}%) – {fmt_bytes(int(speed))}/s")
        return callback

    def _stream_download(self, task: DownloadTask, url: str, filepath: Path):
        """Stream download with progress reporting and resume support."""
        import requests

        headers = {}
        if self.token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {self.token}"

        # Resume support: check existing file
        resume_pos = 0
        if filepath.exists():
            resume_pos = filepath.stat().st_size
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
                self.log_signal.emit(task.id, f"Resuming from {self._format_bytes(resume_pos)}")

        proxy = self.settings.get("proxy", "").strip()
        proxies = {"http": proxy, "https": proxy} if proxy else None

        try:
            resp = requests.get(url, headers=headers, stream=True,
                                proxies=proxies, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 416:
                # File already complete
                self.log_signal.emit(task.id, "File already complete (416).")
                return True, ""
            return False, str(e)
        except Exception as e:
            return False, str(e)

        total = int(resp.headers.get("Content-Length", 0)) + resume_pos
        mode = "ab" if resume_pos > 0 else "wb"

        chunk_size = self.settings.get("chunk_size_mb", 8) * 1024 * 1024
        downloaded = resume_pos
        start_time = time.time()
        last_update = start_time

        try:
            with open(filepath, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if self._stop.is_set():
                        return False, "Cancelled"
                    while self._pause.is_set():
                        time.sleep(0.5)
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        elapsed = now - start_time
                        speed = downloaded / max(elapsed, 0.001)
                        remaining = (total - downloaded) / max(speed, 0.001) if speed > 0 else 0
                        pct = (downloaded / total * 100) if total > 0 else 0

                        if now - last_update >= 0.5:
                            self.progress_signal.emit(
                                task.id, pct, speed,
                                int(remaining), downloaded, total
                            )
                            last_update = now

            return True, ""
        except Exception as e:
            return False, str(e)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_size(s: str) -> int:
        """Parse '5MiB', '1GiB', '500KiB' etc. to bytes."""
        s = s.strip().upper()
        try:
            for suffix, mult in [
                ("GIB", 1024**3), ("MIB", 1024**2), ("KIB", 1024),
                ("GB", 1000**3),  ("MB", 1000**2),  ("KB", 1000),
                ("B", 1),
            ]:
                if s.endswith(suffix):
                    return int(float(s[:-len(suffix)]) * mult)
            return int(s.rstrip("B"))
        except Exception:
            return 0

    @staticmethod
    def _parse_eta(s: str) -> int:
        """Parse '3m30s', '1h5m' etc. to seconds."""
        s = s.lower()
        total = 0
        for match in re.finditer(r"(\d+)([hms])", s):
            v, u = int(match.group(1)), match.group(2)
            total += v * {"h": 3600, "m": 60, "s": 1}.get(u, 0)
        return total

    @staticmethod
    def _format_bytes(b: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"


# ══════════════════════════════════════════════════════════════════════════════
#  CORE DOWNLOAD MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class DownloadManager:
    """
    Orchestrates the download queue, spawning DownloadWorker threads
    up to the parallel download limit.
    """

    def __init__(self, settings: SettingsManager, queue: QueueManager,
                 aria2: Aria2Manager, token_mgr: TokenManager):
        self.settings   = settings
        self.queue      = queue
        self.aria2      = aria2
        self.token_mgr  = token_mgr
        self._workers: Dict[str, DownloadWorker] = {}
        self._callbacks = []  # list of callback functions for UI updates

    def add_callback(self, fn):
        self._callbacks.append(fn)

    def _notify(self, event: str, *args):
        for cb in self._callbacks:
            try:
                cb(event, *args)
            except Exception as e:
                log.error(f"Callback error: {e}")

    def add_task(self, task: DownloadTask):
        self.queue.add(task)
        self._notify("task_added", task)
        self._try_start_next()

    def _try_start_next(self):
        max_parallel = self.settings.get("parallel_downloads", 2)
        while self.queue.active_count() < max_parallel:
            task = self.queue.next_queued()
            if not task:
                break
            self._start_worker(task)

    def _start_worker(self, task: DownloadTask):
        token = self.token_mgr.load_token()
        worker = DownloadWorker(task, self.settings, self.aria2, token)
        worker.progress_signal.connect(self._on_progress)
        worker.status_signal.connect(self._on_status)
        worker.log_signal.connect(self._on_log)
        worker.finished_signal.connect(self._on_finished)
        self._workers[task.id] = worker
        task.status = DownloadStatus.ACTIVE.value
        self.queue.update(task)
        worker.start()
        self._notify("task_started", task)

    def _on_progress(self, task_id, pct, speed, eta, downloaded, total):
        task = self.queue.get(task_id)
        if task:
            task.progress   = pct
            task.speed      = speed
            task.eta        = eta
            task.downloaded = downloaded
            task.size       = total
            self.queue.update(task)
            self._notify("progress", task)

    def _on_status(self, task_id, status):
        task = self.queue.get(task_id)
        if task:
            task.status = status
            self.queue.update(task)
            self._notify("status", task)

    def _on_log(self, task_id, msg):
        self._notify("log", task_id, msg)

    def _on_finished(self, task_id, success, error):
        task = self.queue.get(task_id)
        if task:
            task.status      = DownloadStatus.COMPLETE.value if success else DownloadStatus.FAILED.value
            task.error       = error
            task.finished_at = datetime.now().isoformat()
            self.queue.update(task)
            self._notify("finished", task)
        self._workers.pop(task_id, None)
        self._try_start_next()
        # Auto-shutdown
        if self.settings.get("auto_shutdown") and not self.queue.active_count():
            if not self.queue.next_queued():
                self._notify("all_done")

    def pause_task(self, task_id: str):
        w = self._workers.get(task_id)
        if w:
            w.pause()
            task = self.queue.get(task_id)
            if task:
                task.status = DownloadStatus.PAUSED.value
                self.queue.update(task)
                self._notify("status", task)

    def resume_task(self, task_id: str):
        task = self.queue.get(task_id)
        if not task:
            return
        w = self._workers.get(task_id)
        if w:
            w.resume_dl()
            task.status = DownloadStatus.ACTIVE.value
            self.queue.update(task)
            self._notify("status", task)
        else:
            task.status = DownloadStatus.QUEUED.value
            self.queue.update(task)
            self._try_start_next()

    def cancel_task(self, task_id: str):
        w = self._workers.get(task_id)
        if w:
            w.stop()

    def retry_task(self, task_id: str):
        task = self.queue.get(task_id)
        if task:
            task.status   = DownloadStatus.QUEUED.value
            task.retries  = 0
            task.progress = 0.0
            task.error    = ""
            self.queue.update(task)
            self._try_start_next()

    def pause_all(self):
        for tid in list(self._workers.keys()):
            self.pause_task(tid)

    def resume_all(self):
        for task in self.queue.tasks:
            if task.status in (DownloadStatus.PAUSED.value, DownloadStatus.QUEUED.value):
                self.resume_task(task.id)


# ══════════════════════════════════════════════════════════════════════════════
#  GGUF KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
GGUF_QUANTS = {
    "Q2_K": {
        "quality": "Very Low",
        "size_mult": 0.28,
        "ram": "Minimal",
        "speed": "Very Fast",
        "desc": "Extremely small. Significant quality loss. Only for very low RAM (<4 GB) systems.",
        "recommend": "⚠️ Not recommended unless you have <4 GB RAM"
    },
    "Q3_K_S": {
        "quality": "Low",
        "size_mult": 0.34,
        "ram": "Very Low",
        "speed": "Very Fast",
        "desc": "Small size, noticeable quality degradation. Still usable for simple tasks.",
        "recommend": "For 4 GB RAM systems only"
    },
    "Q3_K_M": {
        "quality": "Low-Medium",
        "size_mult": 0.36,
        "ram": "Low",
        "speed": "Fast",
        "desc": "Better than S variant, still compact. Acceptable for casual use.",
        "recommend": "4-6 GB RAM"
    },
    "Q4_K_S": {
        "quality": "Medium",
        "size_mult": 0.43,
        "ram": "Medium",
        "speed": "Fast",
        "desc": "Good balance of size and quality. Popular choice.",
        "recommend": "6-8 GB RAM"
    },
    "Q4_K_M": {
        "quality": "Medium-Good",
        "size_mult": 0.45,
        "ram": "Medium",
        "speed": "Fast",
        "desc": "⭐ Most popular quantization. Excellent size/quality balance. Recommended for most users.",
        "recommend": "⭐ Best for 8 GB RAM — MOST RECOMMENDED"
    },
    "Q5_K_S": {
        "quality": "Good",
        "size_mult": 0.53,
        "ram": "Medium-High",
        "speed": "Medium",
        "desc": "Very good quality. Slightly larger than Q4.",
        "recommend": "10-12 GB RAM"
    },
    "Q5_K_M": {
        "quality": "Very Good",
        "size_mult": 0.55,
        "ram": "High",
        "speed": "Medium",
        "desc": "Excellent quality with manageable size. Great for demanding tasks.",
        "recommend": "12-16 GB RAM"
    },
    "Q6_K": {
        "quality": "Excellent",
        "size_mult": 0.62,
        "ram": "High",
        "speed": "Medium-Slow",
        "desc": "Near-perfect quality. Only slightly larger than Q5.",
        "recommend": "16+ GB RAM"
    },
    "Q8_0": {
        "quality": "Near-Perfect",
        "size_mult": 0.83,
        "ram": "Very High",
        "speed": "Slow",
        "desc": "Almost indistinguishable from full precision. Very large files.",
        "recommend": "24+ GB RAM"
    },
    "IQ4_XS": {
        "quality": "Medium-Good",
        "size_mult": 0.42,
        "ram": "Medium",
        "speed": "Fast",
        "desc": "Improved quantization (IQ series). Better quality than Q4_K_S at similar size.",
        "recommend": "8 GB RAM — Good alternative to Q4_K_M"
    },
    "IQ3_M": {
        "quality": "Low-Medium",
        "size_mult": 0.35,
        "ram": "Low",
        "speed": "Fast",
        "desc": "Modern IQ quantization at 3-bit. Better than Q3_K_M at same size.",
        "recommend": "4-6 GB RAM"
    },
    "F16": {
        "quality": "Perfect",
        "size_mult": 1.0,
        "ram": "Extreme",
        "speed": "Very Slow",
        "desc": "Full 16-bit precision. Huge files, maximum quality. Mostly for developers.",
        "recommend": "64+ GB RAM — For developers only"
    },
}

HELP_ARTICLES = {
    "What is GGUF?": """
GGUF (GPT-Generated Unified Format) is a file format for storing large language models (LLMs).

It was created by the llama.cpp project to store AI models in a single, portable file.

Key features:
• Single file contains the entire model
• Includes metadata (architecture, tokenizer, hyperparameters)
• Supports quantization (size reduction)
• Works on CPU, GPU, and Apple Silicon
• Used by llama.cpp, LM Studio, Ollama, and more

Why GGUF?
Before GGUF, models needed multiple files in different formats. GGUF simplifies everything into one file that works everywhere.
""",
    "What is Quantization?": """
Quantization is a technique to reduce AI model file sizes by using fewer bits to store numbers.

Think of it like this:
• Full precision (F32): 32 bits per number → exact, but huge
• Half precision (F16): 16 bits per number → still good, half the size
• Q8_0: 8 bits per number → very good, 1/4 the size
• Q4_K_M: 4 bits per number → good enough, 1/8 the size
• Q2_K: 2 bits per number → smallest, but quality suffers

The "K" in names like Q4_K_M means "K-quant" — a smarter quantization method that applies different precision to different parts of the model, preserving quality better than simple quantization.

Which to choose?
→ 8 GB RAM: Q4_K_M (best balance)
→ 16 GB RAM: Q5_K_M or Q6_K
→ 24+ GB RAM: Q8_0
""",
    "Why Downloads Fail?": """
Large file downloads fail for many reasons:

1. Wi-Fi instability
   Campus/public Wi-Fi often drops connections during large transfers.

2. Server timeouts
   HuggingFace servers may disconnect idle connections.

3. Rate limiting
   Without a HF token, you may be rate-limited.

4. Power/sleep interruptions
   Laptop sleep can kill network connections.

5. Disk space
   Running out of disk space mid-download.

6. ISP issues
   Some ISPs throttle or block large file downloads.

How this app helps:
✓ Automatic retry on failure
✓ Resume from where it left off
✓ aria2c uses multiple connections (harder to fail)
✓ Persistent queue survives app restarts
""",
    "How Resume Works?": """
Resume support means you can:
• Pause a download and continue later
• Close the app and restart without losing progress
• Survive Wi-Fi disconnections
• Survive power outages (if temp files are preserved)

How it works technically:
1. The app tracks how many bytes have been downloaded
2. On resume, it sends a "Range: bytes=X-" header
3. The server sends only the missing bytes
4. The file is assembled from the existing part + new bytes

aria2c method:
• Creates a .aria2 control file alongside the download
• This file tracks which segments have been downloaded
• On restart, aria2c reads the control file and resumes

Stream download method:
• Checks the existing file size
• Sends Range header to continue from that position

Important: Do NOT delete .aria2 files while downloading — they contain resume information!
""",
    "Which Quant Should I Choose?": """
Quick guide based on your RAM:

Available RAM → Recommended Quant:

4 GB  → Q2_K (barely works for 7B models)
6 GB  → Q3_K_M (7B models OK)
8 GB  → Q4_K_M ⭐ (best choice for 7B models)
12 GB → Q5_K_M (7B excellent, 13B OK)
16 GB → Q6_K (7B/13B excellent)
24 GB → Q8_0 (13B excellent)
32 GB → Q8_0 (20B-34B models)
64 GB → F16 (full precision)

Model size reference (7B model):
• Q4_K_M ≈ 4.1 GB
• Q5_K_M ≈ 5.0 GB
• Q8_0   ≈ 7.7 GB

For 13B model, multiply by ~1.85
For 34B model, multiply by ~4.8
For 70B model, multiply by ~10

Rule of thumb:
Choose the highest quant that leaves 2-4 GB RAM free for the OS.
""",
    "What is MoE?": """
MoE = Mixture of Experts

A MoE model is a type of AI architecture where the model has multiple "expert" sub-networks. For each input, only some experts are activated.

Example:
• Mixtral 8x7B: 8 experts of 7B each, but only 2 activate per token
• Total parameters: 46.7B
• Active parameters: ~12.9B
• Performance: Much better than a regular 13B model

Why does this matter for GGUF?
• MoE models are MUCH larger files than regular models
• But they're faster to run than their parameter count suggests
• RAM requirement is based on ACTIVE experts, not total
• Still need to load ALL experts into RAM though

Example RAM requirements for Mixtral 8x7B GGUF:
• Q4_K_M: ~26 GB RAM
• Q5_K_M: ~33 GB RAM
• Q8_0:   ~47 GB RAM
""",
    "HuggingFace Token Guide": """
A HuggingFace token is like a password that identifies you to the HF servers.

Why do you need one?
• Higher download rate limits
• Access to private/gated models
• Better download reliability
• Required for some models

How to get a free token:
1. Go to https://huggingface.co
2. Create a free account
3. Go to Settings → Access Tokens
4. Click "New token"
5. Select "Read" permission
6. Copy the token (starts with hf_)
7. Paste it in this app's Login tab

Is it safe?
• Your token is stored locally on your computer only
• It is obfuscated (not plain text) in storage
• It is only sent to HuggingFace servers
• You can revoke it anytime from HF settings

Note: Never share your token with anyone!
""",
}


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE SHEET
# ══════════════════════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #0d1117;
    color: #e6edf3;
}

QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: "Segoe UI", "SF Pro Display", Arial, sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #161b22;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #21262d;
    color: #8b949e;
    padding: 10px 20px;
    border: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #161b22;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

QTabBar::tab:hover {
    background-color: #30363d;
    color: #e6edf3;
}

QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}

QPushButton:pressed {
    background-color: #388bfd1a;
}

QPushButton#primary {
    background-color: #238636;
    border-color: #2ea043;
    color: #ffffff;
    font-weight: 700;
    padding: 10px 20px;
    font-size: 14px;
}

QPushButton#primary:hover {
    background-color: #2ea043;
}

QPushButton#danger {
    background-color: #da3633;
    border-color: #f85149;
    color: #ffffff;
}

QPushButton#danger:hover {
    background-color: #f85149;
}

QPushButton#accent {
    background-color: #1f6feb;
    border-color: #388bfd;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#accent:hover {
    background-color: #388bfd;
}

QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0d1117;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #264f78;
}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #58a6ff;
    outline: none;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #8b949e;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #21262d;
    color: #e6edf3;
}

QProgressBar {
    background-color: #21262d;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:1 #388bfd);
    border-radius: 4px;
}

QScrollBar:vertical {
    background-color: #0d1117;
    width: 8px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 8px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 4px;
}

QListWidget, QTreeWidget, QTableWidget {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    alternate-background-color: #0d1117;
    color: #e6edf3;
}

QListWidget::item, QTreeWidget::item, QTableWidget::item {
    padding: 6px;
    border-radius: 4px;
}

QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
    background-color: #21262d;
    color: #58a6ff;
}

QListWidget::item:hover, QTreeWidget::item:hover {
    background-color: #21262d;
}

QHeaderView::section {
    background-color: #21262d;
    color: #8b949e;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #30363d;
    font-weight: 600;
}

QGroupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
    color: #8b949e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #8b949e;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QCheckBox {
    color: #e6edf3;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #30363d;
    border-radius: 4px;
    background-color: #21262d;
}

QCheckBox::indicator:checked {
    background-color: #238636;
    border-color: #2ea043;
}

QLabel#header {
    color: #58a6ff;
    font-size: 20px;
    font-weight: 700;
}

QLabel#subheader {
    color: #8b949e;
    font-size: 13px;
}

QLabel#status-ok {
    color: #3fb950;
}

QLabel#status-warn {
    color: #d29922;
}

QLabel#status-err {
    color: #f85149;
}

QFrame#separator {
    background-color: #30363d;
    max-height: 1px;
}

QToolTip {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #30363d;
}

QMenuBar {
    background-color: #161b22;
    color: #e6edf3;
    border-bottom: 1px solid #30363d;
}

QMenuBar::item:selected {
    background-color: #21262d;
}

QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #e6edf3;
}

QMenu::item:selected {
    background-color: #21262d;
}

QSplitter::handle {
    background-color: #30363d;
}
"""

LIGHT_STYLE = """
QMainWindow, QDialog, QWidget {
    background-color: #f6f8fa;
    color: #24292f;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #d0d7de;
    background-color: #ffffff;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #f6f8fa;
    color: #57606a;
    padding: 10px 20px;
    border: 1px solid #d0d7de;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #0969da;
}

QPushButton {
    background-color: #f6f8fa;
    color: #24292f;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #eaeef2;
}

QPushButton#primary {
    background-color: #2da44e;
    color: white;
    border-color: #2da44e;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 20px;
}

QPushButton#accent {
    background-color: #0969da;
    color: white;
    font-weight: 600;
}

QLineEdit, QTextEdit, QSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 8px 12px;
    color: #24292f;
}

QProgressBar {
    background-color: #eaeef2;
    border: none;
    border-radius: 4px;
    height: 8px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0969da, stop:1 #218bff);
    border-radius: 4px;
}

QLabel#header { color: #0969da; font-size: 20px; font-weight: 700; }
QLabel#status-ok { color: #1a7f37; }
QLabel#status-warn { color: #9a6700; }
QLabel#status-err { color: #cf222e; }

QGroupBox {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #57606a;
    font-size: 12px;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
def make_label(text: str, style_id: str = "", size: int = 0, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    if style_id:
        lbl.setObjectName(style_id)
    if size:
        f = lbl.font()
        f.setPointSize(size)
        lbl.setFont(f)
    if bold:
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
    return lbl


def make_separator() -> QFrame:
    sep = QFrame()
    sep.setObjectName("separator")
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFixedHeight(1)
    return sep


def fmt_bytes(b: int) -> str:
    """Format bytes to human-readable string."""
    if b <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def fmt_eta(sec: int) -> str:
    if sec <= 0:
        return "--:--"
    td = timedelta(seconds=sec)
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def make_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8]


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD CARD WIDGET
# ══════════════════════════════════════════════════════════════════════════════
class DownloadCard(QFrame):
    """Visual card representing one download in the queue."""

    pause_requested  = Signal(str)
    resume_requested = Signal(str)
    cancel_requested = Signal(str)
    retry_requested  = Signal(str)
    remove_requested = Signal(str)

    STATUS_COLORS = {
        DownloadStatus.QUEUED.value:    "#d29922",
        DownloadStatus.ACTIVE.value:    "#58a6ff",
        DownloadStatus.PAUSED.value:    "#8b949e",
        DownloadStatus.COMPLETE.value:  "#3fb950",
        DownloadStatus.FAILED.value:    "#f85149",
        DownloadStatus.VERIFYING.value: "#a371f7",
        DownloadStatus.RETRYING.value:  "#d29922",
    }

    def __init__(self, task: DownloadTask, parent=None):
        super().__init__(parent)
        self.task = task
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin: 3px 0;
            }
            QFrame:hover { border-color: #58a6ff; }
        """)
        self._build_ui()
        self.update_display(task)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Top row: filename + status badge
        top = QHBoxLayout()
        self.lbl_name   = QLabel("filename")
        self.lbl_name.setStyleSheet("font-weight: 700; font-size: 14px; color: #e6edf3;")
        self.lbl_name.setWordWrap(False)
        self.lbl_status = QLabel("Status")
        self.lbl_status.setStyleSheet(
            "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600;"
        )
        top.addWidget(self.lbl_name, 1)
        top.addWidget(self.lbl_status)
        layout.addLayout(top)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Info row: speed, eta, downloaded/total
        info = QHBoxLayout()
        self.lbl_speed = QLabel("Speed: --")
        self.lbl_speed.setStyleSheet("color: #8b949e; font-size: 12px;")
        self.lbl_eta   = QLabel("ETA: --")
        self.lbl_eta.setStyleSheet("color: #8b949e; font-size: 12px;")
        self.lbl_size  = QLabel("0 B / 0 B")
        self.lbl_size.setStyleSheet("color: #8b949e; font-size: 12px;")
        self.lbl_pct   = QLabel("0%")
        self.lbl_pct.setStyleSheet("color: #58a6ff; font-weight: 600; font-size: 12px;")
        info.addWidget(self.lbl_speed)
        info.addWidget(self.lbl_eta)
        info.addWidget(self.lbl_size, 1)
        info.addWidget(self.lbl_pct)
        layout.addLayout(info)

        # Error label (hidden by default)
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #f85149; font-size: 12px;")
        self.lbl_error.setVisible(False)
        layout.addWidget(self.lbl_error)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_pause  = QPushButton("⏸ Pause")
        self.btn_resume = QPushButton("▶ Resume")
        self.btn_cancel = QPushButton("✕ Cancel")
        self.btn_retry  = QPushButton("↺ Retry")
        self.btn_remove = QPushButton("🗑 Remove")
        self.btn_cancel.setObjectName("danger")
        self.btn_retry.setObjectName("accent")

        for btn in (self.btn_pause, self.btn_resume, self.btn_cancel,
                    self.btn_retry, self.btn_remove):
            btn.setFixedHeight(28)
            btn_row.addWidget(btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Connect buttons
        self.btn_pause.clicked.connect(lambda: self.pause_requested.emit(self.task.id))
        self.btn_resume.clicked.connect(lambda: self.resume_requested.emit(self.task.id))
        self.btn_cancel.clicked.connect(lambda: self.cancel_requested.emit(self.task.id))
        self.btn_retry.clicked.connect(lambda: self.retry_requested.emit(self.task.id))
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self.task.id))

    def update_display(self, task: DownloadTask):
        self.task = task
        self.lbl_name.setText(task.filename)
        self.lbl_name.setToolTip(task.url)

        status = task.status
        color = self.STATUS_COLORS.get(status, "#8b949e")
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(
            f"border-radius: 4px; padding: 2px 8px; font-size: 11px;"
            f" font-weight: 600; background-color: {color}22; color: {color};"
        )

        self.progress_bar.setValue(int(task.progress))

        speed_str = f"{fmt_bytes(int(task.speed))}/s" if task.speed > 0 else "--"
        self.lbl_speed.setText(f"↓ {speed_str}")
        self.lbl_eta.setText(f"ETA: {fmt_eta(task.eta)}" if task.eta else "ETA: --")
        self.lbl_size.setText(f"{fmt_bytes(task.downloaded)} / {fmt_bytes(task.size)}")
        self.lbl_pct.setText(f"{task.progress:.1f}%")

        if task.error:
            self.lbl_error.setText(f"⚠ {task.error}\n💡 Try clicking the Retry button.")
            self.lbl_error.setStyleSheet("color: #f85149; font-size: 12px; padding: 4px;")
            self.lbl_error.setVisible(True)
        else:
            self.lbl_error.setVisible(False)

        # Show/hide buttons based on status
        active = status == DownloadStatus.ACTIVE.value
        paused = status == DownloadStatus.PAUSED.value
        failed = status == DownloadStatus.FAILED.value
        done   = status == DownloadStatus.COMPLETE.value
        queued = status == DownloadStatus.QUEUED.value

        self.btn_pause.setVisible(active)
        self.btn_resume.setVisible(paused or queued)
        self.btn_cancel.setVisible(active or paused or queued)
        self.btn_retry.setVisible(failed)
        self.btn_remove.setVisible(done or failed)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
class DownloadTab(QWidget):
    """Main download tab — paste URL, configure, download."""

    add_download_requested = Signal(dict)

    def __init__(self, settings: SettingsManager, aria2: Aria2Manager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.aria2    = aria2
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = make_label("⬇  Download Models", "header")
        sub = make_label("Paste a HuggingFace URL or any direct file link below", "subheader")
        layout.addWidget(hdr)
        layout.addWidget(sub)
        layout.addWidget(make_separator())

        # ── URL Input ─────────────────────────────────────────────────────────
        url_grp = QGroupBox("Download URL")
        url_lay = QVBoxLayout(url_grp)

        url_hint = QLabel(
            "💡 Examples:\n"
            "  • https://huggingface.co/TheBloke/Llama-2-7B-GGUF\n"
            "  • https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf\n"
            "  • Direct .gguf URL from any website"
        )
        url_hint.setStyleSheet("color: #8b949e; font-size: 12px;")
        url_lay.addWidget(url_hint)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(
            "Paste HuggingFace repo URL, file URL, or any direct download link..."
        )
        self.url_input.setToolTip(
            "Paste the URL of the model you want to download.\n"
            "For HuggingFace repos, paste the full repo URL and\n"
            "use the 'Browse Files' button to pick specific files."
        )
        self.url_input.setMinimumHeight(40)
        url_lay.addWidget(self.url_input)

        url_btns = QHBoxLayout()
        self.btn_paste   = QPushButton("📋 Paste from Clipboard")
        self.btn_browse  = QPushButton("🔍 Browse Repo Files")
        self.btn_paste.setToolTip("Paste URL from your clipboard")
        self.btn_browse.setToolTip("Enter a repo ID and browse available files")
        url_btns.addWidget(self.btn_paste)
        url_btns.addWidget(self.btn_browse)
        url_btns.addStretch()
        url_lay.addLayout(url_btns)
        layout.addWidget(url_grp)

        # ── Repo browser (collapsible) ─────────────────────────────────────────
        self.repo_grp = QGroupBox("Repository File Browser")
        self.repo_grp.setVisible(False)
        repo_lay = QVBoxLayout(self.repo_grp)

        repo_input_row = QHBoxLayout()
        self.repo_input = QLineEdit()
        self.repo_input.setPlaceholderText("e.g. TheBloke/Llama-2-7B-GGUF")
        self.repo_input.setToolTip("Enter the Hugging Face repo ID (user/model-name)")
        self.btn_list_files = QPushButton("List Files")
        self.btn_list_files.setObjectName("accent")
        repo_input_row.addWidget(QLabel("Repo ID:"))
        repo_input_row.addWidget(self.repo_input, 1)
        repo_input_row.addWidget(self.btn_list_files)
        repo_lay.addLayout(repo_input_row)

        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(["Filename", "Size", "Type", "Select"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_table.setMaximumHeight(200)
        repo_lay.addWidget(self.file_table)

        self.btn_use_selected = QPushButton("⬇ Download Selected File")
        self.btn_use_selected.setObjectName("primary")
        repo_lay.addWidget(self.btn_use_selected)
        layout.addWidget(self.repo_grp)

        # ── Settings ──────────────────────────────────────────────────────────
        cfg_grp = QGroupBox("Download Configuration")
        cfg_lay = QGridLayout(cfg_grp)
        cfg_lay.setColumnStretch(1, 1)
        cfg_lay.setColumnStretch(3, 1)

        # Row 0: Destination + method
        cfg_lay.addWidget(QLabel("Save to:"), 0, 0)
        dest_row = QHBoxLayout()
        self.dest_input = QLineEdit(self.settings.get("download_dir"))
        self.dest_input.setToolTip("Folder where downloaded files will be saved")
        self.btn_dest = QPushButton("Browse...")
        dest_row.addWidget(self.dest_input)
        dest_row.addWidget(self.btn_dest)
        cfg_lay.addLayout(dest_row, 0, 1)

        cfg_lay.addWidget(QLabel("Method:"), 0, 2)
        self.method_combo = QComboBox()
        for m in DownloadMethod:
            self.method_combo.addItem(m.value)
        self.method_combo.setCurrentText(DownloadMethod.AUTO.value)
        self.method_combo.setToolTip(
            "Auto: Best method is chosen automatically\n"
            "HuggingFace Hub: Uses official Python library (reliable, supports private models)\n"
            "aria2c: Multi-connection download tool (faster for large files, resume support)"
        )
        cfg_lay.addWidget(self.method_combo, 0, 3)

        # Row 1: Connections + filename
        cfg_lay.addWidget(QLabel("Connections:"), 1, 0)
        self.conn_spin = QSpinBox()
        self.conn_spin.setRange(1, 32)
        self.conn_spin.setValue(self.settings.get("connections", 16))
        self.conn_spin.setToolTip(
            "Number of parallel connections for aria2c.\n"
            "More connections = faster download but more server load.\n"
            "Recommended: 16 for most connections. Reduce to 4-8 on slow Wi-Fi."
        )
        cfg_lay.addWidget(self.conn_spin, 1, 1)

        cfg_lay.addWidget(QLabel("Filename:"), 1, 2)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Auto-detected from URL")
        self.name_input.setToolTip(
            "Override the filename. Leave blank to auto-detect from URL.\n"
            "Usually you don't need to change this."
        )
        cfg_lay.addWidget(self.name_input, 1, 3)

        layout.addWidget(cfg_grp)

        # ── Method explanation ────────────────────────────────────────────────
        self.method_info = QLabel()
        self.method_info.setStyleSheet(
            "background-color: #161b22; border: 1px solid #30363d;"
            " border-radius: 6px; padding: 10px; color: #8b949e; font-size: 12px;"
        )
        self.method_info.setWordWrap(True)
        self._update_method_info()
        layout.addWidget(self.method_info)

        # ── Download button ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_download = QPushButton("⬇  START DOWNLOAD")
        self.btn_download.setObjectName("primary")
        self.btn_download.setMinimumHeight(50)
        self.btn_download.setToolTip("Add this download to the queue and start downloading")
        btn_row.addWidget(self.btn_download)
        layout.addLayout(btn_row)

        layout.addStretch()

        # ── Connections ────────────────────────────────────────────────────────
        self.btn_paste.clicked.connect(self._paste_url)
        self.btn_browse.clicked.connect(lambda: self.repo_grp.setVisible(not self.repo_grp.isVisible()))
        self.btn_dest.clicked.connect(self._browse_dest)
        self.btn_download.clicked.connect(self._start_download)
        self.btn_list_files.clicked.connect(self._list_repo_files)
        self.btn_use_selected.clicked.connect(self._use_selected_file)
        self.method_combo.currentTextChanged.connect(lambda _: self._update_method_info())

    def _update_method_info(self):
        method = self.method_combo.currentText()
        if method == DownloadMethod.AUTO.value:
            txt = (
                "🤖 Auto Mode: The app will automatically choose the best download method.\n"
                "• HuggingFace repo URLs → uses huggingface_hub library\n"
                "• Direct file URLs → uses aria2c (if available) or streaming download\n"
                "Recommended for most users."
            )
        elif method == DownloadMethod.HF_HUB.value:
            txt = (
                "📦 HuggingFace Hub: Official Python library.\n"
                "✅ Best for: Private/gated models, authentication, small-medium files\n"
                "⚠️ Resume support: Basic (single connection)\n"
                "💡 Requires: huggingface_hub Python package (auto-installed)"
            )
        else:
            aria_ok = "✅ Found" if self.aria2.available else "❌ Not installed (will be downloaded automatically)"
            txt = (
                f"⚡ aria2c: Ultra-reliable multi-connection downloader.\n"
                f"Status: {aria_ok}\n"
                "✅ Best for: Large GGUF files, unstable connections, maximum speed\n"
                "✅ Resume: Full resume support with .aria2 control files\n"
                "✅ Multi-connection: Up to 32 parallel connections"
            )
        self.method_info.setText(txt)

    def _paste_url(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            self.url_input.setText(text)
            # Auto-detect filename from URL
            if not self.name_input.text():
                fname = Path(urllib.parse.urlparse(text).path).name
                if fname:
                    self.name_input.setText(fname)

    def _browse_dest(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Download Folder",
            self.dest_input.text() or str(Path.home())
        )
        if folder:
            self.dest_input.setText(folder)

    def _list_repo_files(self):
        repo_id = self.repo_input.text().strip()
        if not repo_id:
            QMessageBox.warning(self, "Input Required", "Please enter a repository ID.")
            return

        self.btn_list_files.setText("Loading...")
        self.btn_list_files.setEnabled(False)

        # Run in thread to avoid UI freeze
        def worker():
            files = HFApiHelper.list_repo_files(repo_id)
            return files

        thread = threading.Thread(target=self._do_list_files, args=(repo_id,))
        thread.daemon = True
        thread.start()

    def _do_list_files(self, repo_id: str):
        files = HFApiHelper.list_repo_files(repo_id)
        # Update UI in main thread via timer hack
        self._pending_files = files
        self._pending_repo  = repo_id
        QTimer.singleShot(0, self._update_file_table)

    def _update_file_table(self):
        files = getattr(self, "_pending_files", [])
        repo  = getattr(self, "_pending_repo", "")
        self.btn_list_files.setText("List Files")
        self.btn_list_files.setEnabled(True)

        self.file_table.setRowCount(0)
        for f in files:
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            self.file_table.setItem(row, 0, QTableWidgetItem(f["name"]))
            self.file_table.setItem(row, 1, QTableWidgetItem(fmt_bytes(f["size"])))
            ftype = "GGUF" if f["name"].endswith(".gguf") else "Other"
            self.file_table.setItem(row, 2, QTableWidgetItem(ftype))
            btn = QPushButton("Select")
            btn.clicked.connect(lambda _, r=row, ri=repo: self._select_file(r, ri))
            self.file_table.setCellWidget(row, 3, btn)

    def _select_file(self, row: int, repo_id: str):
        fname_item = self.file_table.item(row, 0)
        if fname_item:
            fname = fname_item.text()
            url = HFApiHelper.get_file_url(repo_id, fname)
            self.url_input.setText(url)
            self.name_input.setText(Path(fname).name)
            self.repo_input.setText(repo_id)
            # Store for task creation
            self._selected_repo     = repo_id
            self._selected_hf_file  = fname

    def _use_selected_file(self):
        row = self.file_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Please click 'Select' next to a file first.")
            return
        self._select_file(row, self.repo_input.text())

    def _start_download(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "URL Required", "Please enter or paste a download URL.")
            return

        dest = self.dest_input.text().strip() or self.settings.get("download_dir")
        fname = self.name_input.text().strip()
        if not fname:
            fname = Path(urllib.parse.urlparse(url).path).name or "download"

        method = self.method_combo.currentText()
        conns  = self.conn_spin.value()

        # ---- NEW: Smart Hugging Face URL parsing ----
        repo_id = getattr(self, "_selected_repo", "")
        hf_filename = getattr(self, "_selected_hf_file", "")

        # If no repo_id set from file browser, try to extract from URL
        if not repo_id and "huggingface.co" in url:
            parsed = urllib.parse.urlparse(url)
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(path_parts) >= 2:
                # Check if it's a direct resolve link: /{user}/{repo}/resolve/{ref}/{filename}
                if path_parts[2] == "resolve" and len(path_parts) >= 5:
                    repo_id = f"{path_parts[0]}/{path_parts[1]}"
                    hf_filename = "/".join(path_parts[4:])  # includes subfolders if any
                    fname = path_parts[-1] if not fname else fname
                else:
                    # Could be a repo page or blob link, try to get repo_id
                    repo_id = f"{path_parts[0]}/{path_parts[1]}"
        # ---- End of smart parsing ----

        task_data = {
            "id":          make_id(),
            "url":         url,
            "filename":    fname,
            "dest_dir":    dest,
            "method":      method,
            "connections": conns,
            "max_retries": self.settings.get("max_retries", 10),
            "repo_id":     repo_id,
            "hf_filename": hf_filename,
        }

        self.add_download_requested.emit(task_data)
        self.url_input.clear()
        self.name_input.clear()
        self._selected_repo    = ""
        self._selected_hf_file = ""


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: QUEUE
# ══════════════════════════════════════════════════════════════════════════════
class QueueTab(QWidget):
    """Shows all downloads in the queue with controls."""

    pause_requested  = Signal(str)
    resume_requested = Signal(str)
    cancel_requested = Signal(str)
    retry_requested  = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: Dict[str, DownloadCard] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        hdr = make_label("📋  Download Queue", "header")
        layout.addWidget(hdr)

        # Global controls
        ctrl = QHBoxLayout()
        self.btn_pause_all  = QPushButton("⏸ Pause All")
        self.btn_resume_all = QPushButton("▶ Resume All")
        self.btn_clear_done = QPushButton("🗑 Clear Completed")
        self.lbl_stats      = QLabel("No downloads")
        self.lbl_stats.setStyleSheet("color: #8b949e;")
        ctrl.addWidget(self.btn_pause_all)
        ctrl.addWidget(self.btn_resume_all)
        ctrl.addWidget(self.btn_clear_done)
        ctrl.addStretch()
        ctrl.addWidget(self.lbl_stats)
        layout.addLayout(ctrl)
        layout.addWidget(make_separator())

        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.card_container = QWidget()
        self.card_layout    = QVBoxLayout(self.card_container)
        self.card_layout.setSpacing(6)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.addStretch()
        scroll.setWidget(self.card_container)
        layout.addWidget(scroll)

        self.empty_label = QLabel("No downloads yet.\nGo to the Download tab and add a URL.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #8b949e; font-size: 16px; padding: 40px;")
        self.card_layout.insertWidget(0, self.empty_label)

    def add_task(self, task: DownloadTask):
        card = DownloadCard(task)
        card.pause_requested.connect(self.pause_requested)
        card.resume_requested.connect(self.resume_requested)
        card.cancel_requested.connect(self.cancel_requested)
        card.retry_requested.connect(self.retry_requested)
        card.remove_requested.connect(self._on_remove)
        self._cards[task.id] = card
        # Insert before stretch
        idx = self.card_layout.count() - 1
        self.card_layout.insertWidget(idx, card)
        self.empty_label.setVisible(False)
        self._refresh_stats()

    def update_task(self, task: DownloadTask):
        card = self._cards.get(task.id)
        if card:
            card.update_display(task)
        self._refresh_stats()

    def remove_task(self, task_id: str):
        card = self._cards.pop(task_id, None)
        if card:
            self.card_layout.removeWidget(card)
            card.deleteLater()
        if not self._cards:
            self.empty_label.setVisible(True)
        self._refresh_stats()

    def _on_remove(self, task_id: str):
        self.remove_task(task_id)
        self.remove_requested.emit(task_id)

    def _refresh_stats(self):
        total    = len(self._cards)
        active   = sum(1 for c in self._cards.values() if c.task.status == DownloadStatus.ACTIVE.value)
        complete = sum(1 for c in self._cards.values() if c.task.status == DownloadStatus.COMPLETE.value)
        self.lbl_stats.setText(f"{total} total  •  {active} active  •  {complete} done")

    def clear_completed(self):
        done_ids = [tid for tid, c in self._cards.items()
                    if c.task.status == DownloadStatus.COMPLETE.value]
        for tid in done_ids:
            self.remove_task(tid)
            self.remove_requested.emit(tid)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
class SearchTab(QWidget):
    """Search Hugging Face for GGUF models."""

    download_model_requested = Signal(str, str)  # repo_id, filename

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        make_label("🔍  Search Models", "header")
        hdr = make_label("🔍  Search Models", "header")
        sub = make_label("Search HuggingFace for GGUF models", "subheader")
        layout.addWidget(hdr)
        layout.addWidget(sub)
        layout.addWidget(make_separator())

        # Search bar
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models (e.g. Llama, Mistral, Gemma...)")
        self.search_input.setMinimumHeight(40)
        self.search_input.returnPressed.connect(self._do_search)
        self.cb_gguf_only = QCheckBox("GGUF only")
        self.cb_gguf_only.setChecked(True)
        self.cb_gguf_only.setToolTip("Filter results to GGUF-format models only")
        self.btn_search = QPushButton("🔍 Search")
        self.btn_search.setObjectName("accent")
        self.btn_search.setMinimumHeight(40)
        self.btn_search.clicked.connect(self._do_search)
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.cb_gguf_only)
        search_row.addWidget(self.btn_search)
        layout.addLayout(search_row)

        # Results
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Model ID", "Downloads", "Likes", "Tags", "Action"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table)

        self.lbl_results = QLabel("Enter a search query and click Search.")
        self.lbl_results.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(self.lbl_results)

    def _do_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        self.btn_search.setText("Searching...")
        self.btn_search.setEnabled(False)
        self.results_table.setRowCount(0)
        self.lbl_results.setText("Searching HuggingFace...")

        def worker():
            gguf_only = self.cb_gguf_only.isChecked()
            results = HFApiHelper.search_models(query, filter_gguf=gguf_only)
            self._pending_results = results
            QTimer.singleShot(0, self._show_results)

        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    def _show_results(self):
        results = getattr(self, "_pending_results", [])
        self.btn_search.setText("🔍 Search")
        self.btn_search.setEnabled(True)

        self.results_table.setRowCount(0)
        for m in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(m["id"]))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{m['downloads']:,}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(m["likes"])))
            tags = ", ".join(m["tags"][:5]) if m["tags"] else ""
            self.results_table.setItem(row, 3, QTableWidgetItem(tags))

            btn = QPushButton("Browse")
            btn.setToolTip(f"Open {m['id']} in browser or browse files")
            btn.clicked.connect(lambda _, mid=m["id"]: self._open_model(mid))
            self.results_table.setCellWidget(row, 4, btn)

        self.lbl_results.setText(f"Found {len(results)} models")

    def _open_model(self, repo_id: str):
        url = QUrl(f"https://huggingface.co/{repo_id}")
        QDesktopServices.openUrl(url)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: LOGIN
# ══════════════════════════════════════════════════════════════════════════════
class LoginTab(QWidget):
    """HuggingFace token login tab."""

    token_changed = Signal(str)  # emitted with new token or "" on logout

    def __init__(self, token_mgr: TokenManager, parent=None):
        super().__init__(parent)
        self.token_mgr  = token_mgr
        self._token     = token_mgr.load_token() or ""
        self._user_info = None
        self._build_ui()
        if self._token:
            self._show_logged_in()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        hdr = make_label("🔑  HuggingFace Login", "header")
        sub = make_label(
            "Login with your HuggingFace token to access private models and avoid rate limits.",
            "subheader"
        )
        sub.setWordWrap(True)
        layout.addWidget(hdr)
        layout.addWidget(sub)
        layout.addWidget(make_separator())

        # Status card
        self.status_card = QFrame()
        self.status_card.setStyleSheet(
            "background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px;"
        )
        status_lay = QVBoxLayout(self.status_card)

        self.lbl_login_status = QLabel("Not logged in")
        self.lbl_login_status.setObjectName("status-warn")
        self.lbl_login_status.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.lbl_user_info = QLabel("")
        self.lbl_user_info.setStyleSheet("color: #8b949e;")
        status_lay.addWidget(self.lbl_login_status)
        status_lay.addWidget(self.lbl_user_info)
        layout.addWidget(self.status_card)

        # Token input
        token_grp = QGroupBox("Access Token")
        token_lay = QVBoxLayout(token_grp)

        info_lbl = QLabel(
            "📌 How to get your free token:\n"
            "1. Go to https://huggingface.co\n"
            "2. Sign up for free (takes 30 seconds)\n"
            "3. Go to Settings → Access Tokens\n"
            "4. Create a new token (Read permission is enough)\n"
            "5. Paste it below"
        )
        info_lbl.setStyleSheet("color: #8b949e; font-size: 12px; line-height: 1.6;")
        info_lbl.setWordWrap(True)
        token_lay.addWidget(info_lbl)

        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.token_input.setToolTip(
            "Your HuggingFace access token.\n"
            "Starts with 'hf_'\n"
            "Stored locally on your computer, never sent anywhere except HuggingFace."
        )
        self.token_input.setMinimumHeight(40)
        token_lay.addWidget(self.token_input)

        self.cb_show_token = QCheckBox("Show token")
        self.cb_show_token.toggled.connect(
            lambda c: self.token_input.setEchoMode(
                QLineEdit.EchoMode.Normal if c else QLineEdit.EchoMode.Password
            )
        )
        token_lay.addWidget(self.cb_show_token)

        btn_row = QHBoxLayout()
        self.btn_login   = QPushButton("🔑 Save & Verify Token")
        self.btn_login.setObjectName("primary")
        self.btn_login.setMinimumHeight(44)
        self.btn_logout  = QPushButton("🚪 Logout")
        self.btn_logout.setObjectName("danger")
        self.btn_hf_site = QPushButton("🌐 Open HuggingFace Settings")
        btn_row.addWidget(self.btn_login)
        btn_row.addWidget(self.btn_logout)
        btn_row.addWidget(self.btn_hf_site)
        btn_row.addStretch()
        token_lay.addLayout(btn_row)
        layout.addWidget(token_grp)

        # Benefits info
        benefits = QGroupBox("Why log in?")
        b_lay = QVBoxLayout(benefits)
        b_text = QLabel(
            "✅ Higher download speed limits (no rate limiting)\n"
            "✅ Access to private and gated models\n"
            "✅ Better reliability for large downloads\n"
            "✅ Access to newer models that require agreement\n\n"
            "⚠️  Without a token: You may be rate-limited and some models are inaccessible.\n"
            "🆓  HuggingFace accounts are completely FREE."
        )
        b_text.setStyleSheet("color: #8b949e; font-size: 12px; line-height: 1.8;")
        b_text.setWordWrap(True)
        b_lay.addWidget(b_text)
        layout.addWidget(benefits)

        layout.addStretch()

        self.btn_login.clicked.connect(self._do_login)
        self.btn_logout.clicked.connect(self._do_logout)
        self.btn_hf_site.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens"))
        )

    def _do_login(self):
        token = self.token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "Token Required", "Please paste your HuggingFace token.")
            return
        if not self.token_mgr.validate_token(token):
            QMessageBox.warning(self, "Invalid Token",
                                "Token format appears invalid.\nTokens start with 'hf_'")
            return

        self.btn_login.setText("Verifying...")
        self.btn_login.setEnabled(False)

        def verify():
            valid = HFApiHelper.validate_token(token)
            info  = HFApiHelper.get_user_info(token) if valid else None
            self._pending_login = (valid, token, info)
            QTimer.singleShot(0, self._finish_login)

        t = threading.Thread(target=verify)
        t.daemon = True
        t.start()

    def _finish_login(self):
        valid, token, info = self._pending_login
        self.btn_login.setText("🔑 Save & Verify Token")
        self.btn_login.setEnabled(True)

        if valid:
            self._token = token
            self.token_mgr.save_token(token)
            self._user_info = info
            self._show_logged_in()
            self.token_changed.emit(token)
            QMessageBox.information(self, "Login Successful",
                                    f"✅ Logged in as: {info.get('name', 'Unknown') if info else 'Unknown'}")
        else:
            QMessageBox.warning(self, "Login Failed",
                                "Token validation failed.\n"
                                "Please check your token and try again.\n"
                                "Make sure you have internet connectivity.")

    def _show_logged_in(self):
        name = (self._user_info or {}).get("name", "Unknown User") if self._user_info else "Saved Token"
        self.lbl_login_status.setText(f"✅ Logged in as: {name}")
        self.lbl_login_status.setObjectName("status-ok")
        self.lbl_login_status.setStyleSheet("font-size:16px; font-weight:700; color:#3fb950;")
        self.lbl_user_info.setText("Your token is saved securely. You can now download private models.")
        if self._token:
            self.token_input.setText(self._token)

    def _do_logout(self):
        reply = QMessageBox.question(self, "Confirm Logout",
                                     "Remove saved token and log out?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.token_mgr.delete_token()
            self._token = ""
            self._user_info = None
            self.token_input.clear()
            self.lbl_login_status.setText("Not logged in")
            self.lbl_login_status.setStyleSheet("font-size:16px; font-weight:700; color:#d29922;")
            self.lbl_user_info.setText("")
            self.token_changed.emit("")

    def get_token(self) -> Optional[str]:
        return self._token or None


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: GGUF GUIDE
# ══════════════════════════════════════════════════════════════════════════════
class GGUFGuideTab(QWidget):
    """Interactive GGUF quantization guide and RAM calculator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        hdr = make_label("📖  GGUF & Quantization Guide", "header")
        sub = make_label("Learn about GGUF formats, choose the right quantization, estimate RAM needs.", "subheader")
        layout.addWidget(hdr)
        layout.addWidget(sub)
        layout.addWidget(make_separator())

        # RAM Calculator
        calc_grp = QGroupBox("🧮  RAM Calculator")
        calc_lay = QGridLayout(calc_grp)

        calc_lay.addWidget(QLabel("Model Size (B parameters):"), 0, 0)
        self.model_size_combo = QComboBox()
        for size in ["1B", "3B", "7B", "8B", "13B", "14B", "20B", "30B", "34B", "70B", "72B", "120B"]:
            self.model_size_combo.addItem(size)
        self.model_size_combo.setCurrentText("7B")
        self.model_size_combo.setToolTip("Parameter count of the model")
        calc_lay.addWidget(self.model_size_combo, 0, 1)

        calc_lay.addWidget(QLabel("Quantization:"), 0, 2)
        self.quant_combo = QComboBox()
        for q in GGUF_QUANTS.keys():
            self.quant_combo.addItem(q)
        self.quant_combo.setCurrentText("Q4_K_M")
        calc_lay.addWidget(self.quant_combo, 0, 3)

        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.setObjectName("accent")
        self.btn_calc.clicked.connect(self._calculate)
        calc_lay.addWidget(self.btn_calc, 0, 4)

        self.calc_result = QLabel("")
        self.calc_result.setStyleSheet(
            "background:#161b22; border:1px solid #30363d; border-radius:6px;"
            " padding:10px; color:#e6edf3; font-size:13px;"
        )
        self.calc_result.setWordWrap(True)
        calc_lay.addWidget(self.calc_result, 1, 0, 1, 5)
        layout.addWidget(calc_grp)

        # Quant table
        table_grp = QGroupBox("Quantization Comparison Table")
        t_lay = QVBoxLayout(table_grp)

        self.quant_table = QTableWidget(len(GGUF_QUANTS), 6)
        self.quant_table.setHorizontalHeaderLabels(
            ["Quant", "Quality", "Size (7B model)", "Speed", "RAM", "Recommendation"]
        )
        self.quant_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.quant_table.setAlternatingRowColors(True)
        self.quant_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        for row, (qname, qdata) in enumerate(GGUF_QUANTS.items()):
            # Size estimate for 7B model: ~3.5 GB base * mult
            base_7b = 3.5 * 1024**3
            size_est = fmt_bytes(int(base_7b * qdata["size_mult"]))

            self.quant_table.setItem(row, 0, QTableWidgetItem(qname))
            self.quant_table.setItem(row, 1, QTableWidgetItem(qdata["quality"]))
            self.quant_table.setItem(row, 2, QTableWidgetItem(size_est))
            self.quant_table.setItem(row, 3, QTableWidgetItem(qdata["speed"]))
            self.quant_table.setItem(row, 4, QTableWidgetItem(qdata["ram"]))
            self.quant_table.setItem(row, 5, QTableWidgetItem(qdata["recommend"]))

            if "RECOMMENDED" in qdata["recommend"].upper() or "⭐" in qdata["recommend"]:
                for col in range(6):
                    item = self.quant_table.item(row, col)
                    if item:
                        item.setBackground(QColor("#1f3a1f"))

        t_lay.addWidget(self.quant_table)
        layout.addWidget(table_grp)

        # Detail view
        self.detail_lbl = QLabel("Click a row to see detailed information.")
        self.detail_lbl.setStyleSheet(
            "background:#161b22; border:1px solid #30363d; border-radius:6px;"
            " padding:12px; color:#8b949e;"
        )
        self.detail_lbl.setWordWrap(True)
        layout.addWidget(self.detail_lbl)

        self.quant_table.itemSelectionChanged.connect(self._show_quant_detail)

    def _calculate(self):
        size_str = self.model_size_combo.currentText()
        quant    = self.quant_combo.currentText()
        try:
            params_b = float(size_str.rstrip("Bb"))
        except ValueError:
            return

        qdata = GGUF_QUANTS.get(quant)
        if not qdata:
            return

        # Approximate file size: params * bytes per param
        # Full F16: 2 bytes per param, F32: 4 bytes, Q4_K_M: ~0.5 bytes
        bits_map = {
            "Q2_K": 2.5, "Q3_K_S": 3.0, "Q3_K_M": 3.35, "Q4_K_S": 4.5, "Q4_K_M": 4.85,
            "Q5_K_S": 5.5, "Q5_K_M": 5.65, "Q6_K": 6.56, "Q8_0": 8.5,
            "IQ4_XS": 4.25, "IQ3_M": 3.7, "F16": 16.0,
        }
        bits = bits_map.get(quant, 4.85)
        file_size_gb = (params_b * 1e9 * bits / 8) / 1024**3
        ram_needed_gb = file_size_gb * 1.15  # ~15% overhead

        rec = "✅ Should run well" if ram_needed_gb < 8 else (
              "⚠️ May be tight" if ram_needed_gb < 16 else
              "❌ Needs a lot of RAM")

        self.calc_result.setText(
            f"📦 File size (estimate): {file_size_gb:.1f} GB\n"
            f"💾 RAM needed (estimate): {ram_needed_gb:.1f} GB\n"
            f"💡 Quality: {qdata['quality']} | Speed: {qdata['speed']}\n"
            f"🏷️  {qdata['recommend']}\n"
            f"{'─'*40}\n"
            f"ℹ️  {qdata['desc']}"
        )

    def _show_quant_detail(self):
        row = self.quant_table.currentRow()
        if row < 0:
            return
        qname = self.quant_table.item(row, 0).text()
        qdata = GGUF_QUANTS.get(qname)
        if not qdata:
            return
        self.detail_lbl.setText(
            f"📌 {qname}\n\n"
            f"Quality:  {qdata['quality']}\n"
            f"Speed:    {qdata['speed']}\n"
            f"RAM:      {qdata['ram']}\n\n"
            f"{qdata['desc']}\n\n"
            f"🏷️  {qdata['recommend']}"
        )
        self.detail_lbl.setStyleSheet(
            "background:#1a2332; border:1px solid #58a6ff; border-radius:6px;"
            " padding:12px; color:#e6edf3;"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: HELP
# ══════════════════════════════════════════════════════════════════════════════
class HelpTab(QWidget):
    """Built-in help and documentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Article list (sidebar)
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background:#161b22; border-right:1px solid #30363d;")
        s_lay = QVBoxLayout(sidebar)
        s_lay.setContentsMargins(10, 16, 10, 10)

        s_lay.addWidget(make_label("📚 Help Articles", bold=True))
        s_lay.addWidget(make_separator())

        self.article_list = QListWidget()
        self.article_list.setStyleSheet(
            "QListWidget { background:transparent; border:none; }"
            "QListWidget::item { padding:10px 8px; border-radius:6px; }"
            "QListWidget::item:selected { background:#21262d; color:#58a6ff; }"
            "QListWidget::item:hover { background:#1c2128; }"
        )
        for title in HELP_ARTICLES:
            self.article_list.addItem(title)
        s_lay.addWidget(self.article_list, 1)
        layout.addWidget(sidebar)

        # Article content
        content = QWidget()
        c_lay = QVBoxLayout(content)
        c_lay.setContentsMargins(20, 20, 20, 20)

        self.article_title = make_label("Select an article", "header")
        c_lay.addWidget(self.article_title)
        c_lay.addWidget(make_separator())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.article_body = QTextEdit()
        self.article_body.setReadOnly(True)
        self.article_body.setStyleSheet(
            "QTextEdit { background:#0d1117; border:none; color:#e6edf3;"
            " font-size:13px; line-height:1.7; padding:10px; }"
        )
        scroll.setWidget(self.article_body)
        c_lay.addWidget(scroll, 1)
        layout.addWidget(content, 1)

        self.article_list.itemSelectionChanged.connect(self._show_article)

        # Show first article
        if self.article_list.count() > 0:
            self.article_list.setCurrentRow(0)

    def _show_article(self):
        item = self.article_list.currentItem()
        if not item:
            return
        title   = item.text()
        content = HELP_ARTICLES.get(title, "Article not found.")
        self.article_title.setText(f"📄  {title}")
        self.article_body.setPlainText(content.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
class SettingsTab(QWidget):
    """Application settings with tooltips and explanations."""

    settings_changed = Signal()

    def __init__(self, settings: SettingsManager, aria2: Aria2Manager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.aria2    = aria2
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        scroll.setWidget(container)

        hdr = make_label("⚙️  Settings", "header")
        layout.addWidget(hdr)
        layout.addWidget(make_separator())

        # ── Download Settings ─────────────────────────────────────────────────
        dl_grp = QGroupBox("Download Settings")
        dl_lay = QGridLayout(dl_grp)
        dl_lay.setColumnStretch(1, 1)

        # Default download dir
        dl_lay.addWidget(self._lbl_help("Default Download Folder",
            "Where downloaded files are saved by default.\n"
            "You can also change this per-download."), 0, 0)
        dest_row = QHBoxLayout()
        self.dest_input = QLineEdit(self.settings.get("download_dir"))
        self.btn_dest   = QPushButton("Browse")
        self.btn_dest.clicked.connect(self._browse_dest)
        dest_row.addWidget(self.dest_input)
        dest_row.addWidget(self.btn_dest)
        dl_lay.addLayout(dest_row, 0, 1)

        # Connections
        dl_lay.addWidget(self._lbl_help("Connections per Download",
            "Number of parallel connections aria2c uses.\n"
            "More = faster on good connections.\n"
            "Recommended: 16. Reduce to 4-8 on slow/unstable Wi-Fi."), 1, 0)
        self.conn_spin = QSpinBox()
        self.conn_spin.setRange(1, 32)
        self.conn_spin.setValue(self.settings.get("connections", 16))
        dl_lay.addWidget(self.conn_spin, 1, 1)

        # Retries
        dl_lay.addWidget(self._lbl_help("Max Retries",
            "How many times to retry a failed download automatically.\n"
            "Higher = more persistence on bad connections.\n"
            "Recommended: 10"), 2, 0)
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 50)
        self.retry_spin.setValue(self.settings.get("max_retries", 10))
        dl_lay.addWidget(self.retry_spin, 2, 1)

        # Retry wait
        dl_lay.addWidget(self._lbl_help("Retry Wait (seconds)",
            "How long to wait before retrying after a failure.\n"
            "Recommended: 5 seconds"), 3, 0)
        self.retry_wait_spin = QSpinBox()
        self.retry_wait_spin.setRange(1, 60)
        self.retry_wait_spin.setValue(self.settings.get("retry_wait", 5))
        dl_lay.addWidget(self.retry_wait_spin, 3, 1)

        # Parallel downloads
        dl_lay.addWidget(self._lbl_help("Parallel Downloads",
            "How many files to download simultaneously.\n"
            "Recommended: 1-2. Too many = slower speeds per file."), 4, 0)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 8)
        self.parallel_spin.setValue(self.settings.get("parallel_downloads", 2))
        dl_lay.addWidget(self.parallel_spin, 4, 1)

        # Speed limit
        dl_lay.addWidget(self._lbl_help("Speed Limit (KB/s, 0=unlimited)",
            "Limit total download speed in kilobytes per second.\n"
            "0 = no limit (use full connection speed).\n"
            "Example: 5000 = 5 MB/s"), 5, 0)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(0, 100000)
        self.speed_spin.setValue(self.settings.get("speed_limit", 0))
        self.speed_spin.setSuffix(" KB/s")
        dl_lay.addWidget(self.speed_spin, 5, 1)

        # Chunk size
        dl_lay.addWidget(self._lbl_help("Chunk Size (MB)",
            "How much data to read at once during streaming downloads.\n"
            "Larger chunks = slightly more efficient but uses more RAM.\n"
            "Recommended: 8 MB"), 6, 0)
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 64)
        self.chunk_spin.setValue(self.settings.get("chunk_size_mb", 8))
        self.chunk_spin.setSuffix(" MB")
        dl_lay.addWidget(self.chunk_spin, 6, 1)

        layout.addWidget(dl_grp)

        # ── App Settings ──────────────────────────────────────────────────────
        app_grp = QGroupBox("Application Settings")
        app_lay = QGridLayout(app_grp)

        self.cb_dark      = QCheckBox("Dark Mode")
        self.cb_dark.setChecked(self.settings.get("dark_mode", True))
        self.cb_dark.setToolTip("Switch between dark and light theme")
        app_lay.addWidget(self.cb_dark, 0, 0)

        self.cb_tray      = QCheckBox("Minimize to System Tray")
        self.cb_tray.setChecked(self.settings.get("tray_minimize", True))
        self.cb_tray.setToolTip("Keep running in the system tray when you close the window")
        app_lay.addWidget(self.cb_tray, 0, 1)

        self.cb_clipboard = QCheckBox("Auto-detect HF URLs from Clipboard")
        self.cb_clipboard.setChecked(self.settings.get("clipboard_detect", True))
        self.cb_clipboard.setToolTip(
            "Automatically detect when you copy a HuggingFace URL\n"
            "and offer to download it."
        )
        app_lay.addWidget(self.cb_clipboard, 1, 0)

        self.cb_notify    = QCheckBox("Show Notifications")
        self.cb_notify.setChecked(self.settings.get("notifications", True))
        self.cb_notify.setToolTip("Show system notifications when downloads complete or fail")
        app_lay.addWidget(self.cb_notify, 1, 1)

        self.cb_shutdown  = QCheckBox("Auto Shutdown when Queue Empty")
        self.cb_shutdown.setChecked(self.settings.get("auto_shutdown", False))
        self.cb_shutdown.setToolTip("Automatically shut down your computer when all downloads finish")
        app_lay.addWidget(self.cb_shutdown, 2, 0)

        self.cb_verify    = QCheckBox("Verify File Integrity After Download")
        self.cb_verify.setChecked(self.settings.get("verify_hash", True))
        self.cb_verify.setToolTip("Check downloaded files are not corrupted (when hash available)")
        app_lay.addWidget(self.cb_verify, 2, 1)

        layout.addWidget(app_grp)

        # ── Network Settings ──────────────────────────────────────────────────
        net_grp = QGroupBox("Network Settings")
        net_lay = QGridLayout(net_grp)
        net_lay.setColumnStretch(1, 1)

        net_lay.addWidget(self._lbl_help("Proxy (leave blank for none)",
            "HTTP/HTTPS proxy server.\n"
            "Format: http://user:pass@host:port\n"
            "Leave blank if you don't use a proxy."), 0, 0)
        self.proxy_input = QLineEdit(self.settings.get("proxy", ""))
        self.proxy_input.setPlaceholderText("http://proxy:port")
        net_lay.addWidget(self.proxy_input, 0, 1)

        net_lay.addWidget(self._lbl_help("HF Endpoint",
            "HuggingFace API endpoint.\n"
            "Change this if you use a mirror or enterprise HF instance.\n"
            "Default: https://huggingface.co"), 1, 0)
        self.endpoint_input = QLineEdit(self.settings.get("hf_endpoint", "https://huggingface.co"))
        net_lay.addWidget(self.endpoint_input, 1, 1)

        layout.addWidget(net_grp)

        # ── Advanced / aria2 ──────────────────────────────────────────────────
        adv_grp = QGroupBox("Advanced / aria2c")
        adv_lay = QVBoxLayout(adv_grp)

        # aria2 status
        aria_status = "✅ " + self.aria2.get_version() if self.aria2.available else "❌ Not installed"
        adv_lay.addWidget(QLabel(f"aria2c status: {aria_status}"))

        if not self.aria2.available:
            self.btn_install_aria2 = QPushButton("⬇ Download & Install aria2c Automatically")
            self.btn_install_aria2.setObjectName("accent")
            self.btn_install_aria2.clicked.connect(self._install_aria2)
            adv_lay.addWidget(self.btn_install_aria2)

        adv_lay.addWidget(self._lbl_help("Extra aria2c Arguments",
            "Additional command-line arguments passed to aria2c.\n"
            "Only for advanced users. Leave blank for defaults.\n"
            "Example: --seed-time=0 --bt-tracker=..."))
        self.aria2_args = QLineEdit(self.settings.get("aria2_extra_args", ""))
        self.aria2_args.setPlaceholderText("e.g. --seed-time=0")
        adv_lay.addWidget(self.aria2_args)

        layout.addWidget(adv_grp)

        # ── Save/Reset ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_save  = QPushButton("💾 Save Settings")
        self.btn_save.setObjectName("primary")
        self.btn_save.setMinimumHeight(44)
        self.btn_reset = QPushButton("↺ Reset to Defaults")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()

        self.btn_save.clicked.connect(self._save)
        self.btn_reset.clicked.connect(self._reset)

    def _lbl_help(self, text: str, tooltip: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setToolTip(tooltip)
        lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
        return lbl

    def _browse_dest(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Download Folder",
                                                   self.dest_input.text())
        if folder:
            self.dest_input.setText(folder)

    def _save(self):
        self.settings.set("download_dir",        self.dest_input.text())
        self.settings.set("connections",          self.conn_spin.value())
        self.settings.set("max_retries",          self.retry_spin.value())
        self.settings.set("retry_wait",           self.retry_wait_spin.value())
        self.settings.set("parallel_downloads",   self.parallel_spin.value())
        self.settings.set("speed_limit",          self.speed_spin.value() * 1024)
        self.settings.set("chunk_size_mb",        self.chunk_spin.value())
        self.settings.set("dark_mode",            self.cb_dark.isChecked())
        self.settings.set("tray_minimize",        self.cb_tray.isChecked())
        self.settings.set("clipboard_detect",     self.cb_clipboard.isChecked())
        self.settings.set("notifications",        self.cb_notify.isChecked())
        self.settings.set("auto_shutdown",        self.cb_shutdown.isChecked())
        self.settings.set("verify_hash",          self.cb_verify.isChecked())
        self.settings.set("proxy",                self.proxy_input.text())
        self.settings.set("hf_endpoint",          self.endpoint_input.text())
        self.settings.set("aria2_extra_args",     self.aria2_args.text())
        self.settings_changed.emit()
        QMessageBox.information(self, "Saved", "✅ Settings saved successfully!")

    def _reset(self):
        reply = QMessageBox.question(self, "Reset Settings",
                                     "Reset all settings to defaults?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.settings.reset()
            QMessageBox.information(self, "Reset", "Settings reset. Please restart the app.")

    def _install_aria2(self):
        self.btn_install_aria2.setText("Downloading aria2c...")
        self.btn_install_aria2.setEnabled(False)

        def worker():
            ok = self.aria2.download_aria2()
            self._aria2_install_result = ok
            QTimer.singleShot(0, self._aria2_installed)

        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    def _aria2_installed(self):
        ok = getattr(self, "_aria2_install_result", False)
        if ok:
            QMessageBox.information(self, "Success", "✅ aria2c installed successfully!")
            self.btn_install_aria2.setText("✅ aria2c Installed")
        else:
            QMessageBox.warning(self, "Failed",
                                "Could not install aria2c automatically.\n"
                                "Please download it manually from https://github.com/aria2/aria2/releases\n"
                                "and place aria2c.exe in your PATH or in:\n"
                                f"{ARIA2_DIR}")
            self.btn_install_aria2.setText("⬇ Download & Install aria2c Automatically")
            self.btn_install_aria2.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: LOG
# ══════════════════════════════════════════════════════════════════════════════
class LogTab(QWidget):
    """Real-time log viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        hdr_row = QHBoxLayout()
        hdr_row.addWidget(make_label("📋  Activity Log", "header"))
        hdr_row.addStretch()
        self.btn_clear = QPushButton("🗑 Clear")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_open_log = QPushButton("📂 Open Log File")
        self.btn_open_log.clicked.connect(self._open_log_file)
        hdr_row.addWidget(self.btn_clear)
        hdr_row.addWidget(self.btn_open_log)
        layout.addLayout(hdr_row)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace;"
            " font-size: 12px; background:#0d1117; color:#e6edf3; border:1px solid #30363d;"
        )
        layout.addWidget(self.log_view)

    def append(self, task_id: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{ts}] [{task_id}] {msg}")
        # Auto-scroll to bottom
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def _clear(self):
        self.log_view.clear()

    def _open_log_file(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_FILE)))


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP / DEPENDENCY WIZARD DIALOG
# ══════════════════════════════════════════════════════════════════════════════
class SetupWizard(QDialog):
    """
    Shown on first run or when dependencies are missing.
    Automatically installs what's needed.
    """
    # Signal emitted when all checks are done (emitted from worker thread)
    checks_finished = Signal()

    def __init__(self, aria2: Aria2Manager, parent=None):
        super().__init__(parent)
        self.aria2 = aria2
        self.setWindowTitle("HF Download Manager — Setup")
        self.setMinimumSize(580, 440)
        self.setModal(True)
        self._build_ui()

        # Connect the signal AFTER the UI is built
        self.checks_finished.connect(self._show_results)

        # Start the checks after a short delay (ensures dialog is fully shown)
        QTimer.singleShot(500, self._run_checks)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        hdr = make_label("🚀  Welcome to HF Download Manager", "header")
        sub = make_label(
            "Checking and installing required components...\n"
            "This only takes a moment and happens automatically.",
            "subheader"
        )
        sub.setWordWrap(True)
        layout.addWidget(hdr)
        layout.addWidget(sub)
        layout.addWidget(make_separator())

        self.status_area = QTextEdit()
        self.status_area.setReadOnly(True)
        self.status_area.setMinimumHeight(200)
        self.status_area.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 12px;"
            " background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px;"
        )
        layout.addWidget(self.status_area)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.btn_continue = QPushButton("✅ Continue to App")
        self.btn_continue.setObjectName("primary")
        self.btn_continue.setMinimumHeight(44)
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.accept)
        layout.addWidget(self.btn_continue)

    def _log(self, msg: str):
        self.status_area.append(msg)
        cursor = self.status_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.status_area.setTextCursor(cursor)

    def _run_checks(self):
        # Show initial text immediately
        self._log("🔍 Checking Python dependencies...")
        self.progress_bar.setValue(10)

        def worker():
            results = []

            # Check Python deps
            dep_status = DependencyChecker.check_all()
            for pkg, ok in dep_status.items():
                if ok:
                    results.append(("ok", f"  ✅ {pkg} — installed"))
                else:
                    results.append(("warn", f"  ⬇ {pkg} — installing..."))
                    success = DependencyChecker.install(pkg)
                    if success:
                        results.append(("ok", f"  ✅ {pkg} — installed successfully"))
                    else:
                        results.append(("err", f"  ❌ {pkg} — FAILED to install!"))

            results.append(("ok", "\n🔍 Checking aria2c..."))

            if self.aria2.available:
                ver = self.aria2.get_version()
                results.append(("ok", f"  ✅ aria2c — {ver}"))
            else:
                results.append(("warn", "  ⬇ aria2c not found — downloading automatically..."))
                ok = self.aria2.download_aria2()
                if ok:
                    results.append(("ok", f"  ✅ aria2c downloaded and ready!"))
                else:
                    results.append(("warn",
                        "  ⚠️ aria2c could not be auto-downloaded.\n"
                        "     Downloads will use the built-in streaming method.\n"
                        "     You can install aria2c manually later from Settings."
                    ))

            results.append(("ok", "\n✅ Setup complete! Click Continue to start."))

            # Store results and emit signal (safe from any thread)
            self._pending_results = results
            self.checks_finished.emit()        # <-- THIS replaces QTimer.singleShot

        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    def _show_results(self):
        results = getattr(self, "_pending_results", [])
        for kind, msg in results:
            self._log(msg)
        self.progress_bar.setValue(100)
        self.btn_continue.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLIPBOARD MONITOR
# ══════════════════════════════════════════════════════════════════════════════
class ClipboardMonitor(QThread):
    """Polls clipboard for HuggingFace URLs and emits a signal when found."""
    url_detected = Signal(str)

    HF_PATTERN = re.compile(
        r"https?://huggingface\.co/[\w\-./]+"
        r"|https?://hf\.co/[\w\-./]+"
    )

    def __init__(self, interval_ms: int = 1500):
        super().__init__()
        self._last = ""
        self._interval = interval_ms
        self._running  = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            time.sleep(self._interval / 1000)
            try:
                cb = QApplication.clipboard()
                text = cb.text().strip()
                if text != self._last and self.HF_PATTERN.match(text):
                    self._last = text
                    self.url_detected.emit(text)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # ── Core components ───────────────────────────────────────────────────
        self.settings   = SettingsManager()
        self.token_mgr  = TokenManager()
        self.queue_mgr  = QueueManager()
        self.aria2_mgr  = Aria2Manager()
        self.dl_mgr     = DownloadManager(
            self.settings, self.queue_mgr, self.aria2_mgr, self.token_mgr
        )
        self.dl_mgr.add_callback(self._on_download_event)

        # ── Apply theme ───────────────────────────────────────────────────────
        self._apply_theme()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_menu()
        self._build_ui()
        self._build_status_bar()
        self._build_tray()
        self._build_clipboard_monitor()

        # ── Restore queue ─────────────────────────────────────────────────────
        self._restore_queue()

        # ── Status timer ─────────────────────────────────────────────────────
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(2000)

        log.info(f"{APP_NAME} started.")

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        style = DARK_STYLE if self.settings.get("dark_mode", True) else LIGHT_STYLE
        QApplication.instance().setStyleSheet(style)

    # ── Menu Bar ──────────────────────────────────────────────────────────────
    def _build_menu(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")
        act_open_dl = QAction("Open Download Folder", self)
        act_open_dl.triggered.connect(self._open_download_folder)
        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(lambda: self.tabs.setCurrentWidget(self.settings_tab))
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(act_open_dl)
        file_menu.addAction(act_settings)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        # Downloads
        dl_menu = menubar.addMenu("Downloads")
        act_pause_all  = QAction("Pause All", self)
        act_pause_all.triggered.connect(self.dl_mgr.pause_all)
        act_resume_all = QAction("Resume All", self)
        act_resume_all.triggered.connect(self.dl_mgr.resume_all)
        dl_menu.addAction(act_pause_all)
        dl_menu.addAction(act_resume_all)

        # Help
        help_menu = menubar.addMenu("Help")
        act_help = QAction("Help & Documentation", self)
        act_help.triggered.connect(lambda: self.tabs.setCurrentWidget(self.help_tab))
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        act_log = QAction("Open Log File", self)
        act_log.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_FILE))))
        help_menu.addAction(act_help)
        help_menu.addAction(act_log)
        help_menu.addSeparator()
        help_menu.addAction(act_about)

    # ── Main UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # App header bar
        header_bar = QFrame()
        header_bar.setFixedHeight(56)
        header_bar.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            "stop:0 #161b22, stop:1 #0d1117);"
            "border-bottom: 1px solid #30363d;"
        )
        hb_lay = QHBoxLayout(header_bar)
        hb_lay.setContentsMargins(20, 0, 20, 0)

        logo_lbl = QLabel("🤗 HF Download Manager")
        logo_lbl.setStyleSheet("font-size:18px; font-weight:800; color:#58a6ff; letter-spacing:0.5px;")
        version_lbl = QLabel(f"v{APP_VERSION}")
        version_lbl.setStyleSheet("color:#8b949e; font-size:12px;")

        hb_lay.addWidget(logo_lbl)
        hb_lay.addWidget(version_lbl)
        hb_lay.addStretch()

        self.hdr_status = QLabel("● Idle")
        self.hdr_status.setStyleSheet("color:#3fb950; font-size:13px;")
        hb_lay.addWidget(self.hdr_status)

        main_layout.addWidget(header_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Create tabs
        self.download_tab  = DownloadTab(self.settings, self.aria2_mgr)
        self.queue_tab     = QueueTab()
        self.search_tab    = SearchTab()
        self.login_tab     = LoginTab(self.token_mgr)
        self.gguf_tab      = GGUFGuideTab()
        self.help_tab      = HelpTab()
        self.settings_tab  = SettingsTab(self.settings, self.aria2_mgr)
        self.log_tab       = LogTab()

        self.tabs.addTab(self.download_tab,  "⬇ Download")
        self.tabs.addTab(self.queue_tab,     "📋 Queue")
        self.tabs.addTab(self.search_tab,    "🔍 Search")
        self.tabs.addTab(self.login_tab,     "🔑 Login")
        self.tabs.addTab(self.gguf_tab,      "📖 GGUF Guide")
        self.tabs.addTab(self.help_tab,      "❓ Help")
        self.tabs.addTab(self.settings_tab,  "⚙ Settings")
        self.tabs.addTab(self.log_tab,       "📋 Log")

        main_layout.addWidget(self.tabs)

        # ── Wire up signals ───────────────────────────────────────────────────
        self.download_tab.add_download_requested.connect(self._add_download)
        self.queue_tab.pause_requested.connect(self.dl_mgr.pause_task)
        self.queue_tab.resume_requested.connect(self.dl_mgr.resume_task)
        self.queue_tab.cancel_requested.connect(self.dl_mgr.cancel_task)
        self.queue_tab.retry_requested.connect(self.dl_mgr.retry_task)
        self.queue_tab.remove_requested.connect(self._remove_task)
        self.queue_tab.btn_pause_all.clicked.connect(self.dl_mgr.pause_all)
        self.queue_tab.btn_resume_all.clicked.connect(self.dl_mgr.resume_all)
        self.queue_tab.btn_clear_done.clicked.connect(self.queue_tab.clear_completed)
        self.login_tab.token_changed.connect(self._on_token_changed)
        self.settings_tab.settings_changed.connect(self._apply_theme)

    def _build_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Paste a URL in the Download tab to begin.")

    def _build_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(self)
        # Use a simple text icon since we don't have image files
        self.tray.setToolTip(APP_NAME)
        tray_menu = QMenu()
        act_show   = tray_menu.addAction("Show")
        act_quit   = tray_menu.addAction("Quit")
        act_show.triggered.connect(self.show)
        act_quit.triggered.connect(QApplication.instance().quit)
        self.tray.setContextMenu(tray_menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _build_clipboard_monitor(self):
        if not self.settings.get("clipboard_detect", True):
            return
        self.clipboard_mon = ClipboardMonitor()
        self.clipboard_mon.url_detected.connect(self._on_clipboard_url)
        self.clipboard_mon.start()

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.activateWindow()

    # ── Queue management ──────────────────────────────────────────────────────
    def _add_download(self, task_data: dict):
        task = DownloadTask(**{k: v for k, v in task_data.items()
                               if k in DownloadTask.__dataclass_fields__})
        self.dl_mgr.add_task(task)
        self.queue_tab.add_task(task)
        self.tabs.setCurrentWidget(self.queue_tab)
        self.status_bar.showMessage(f"Added: {task.filename}")
        log.info(f"Task added: {task.filename} ({task.url[:60]}...)")

    def _remove_task(self, task_id: str):
        self.queue_mgr.remove(task_id)

    def _restore_queue(self):
        """Re-populate UI from saved queue on startup."""
        for task in self.queue_mgr.tasks:
            self.queue_tab.add_task(task)
        # Re-start any queued downloads
        self.dl_mgr._try_start_next()

    # ── Download event callback ───────────────────────────────────────────────
    def _on_download_event(self, event: str, *args):
        """Called from DownloadManager — must dispatch to UI thread."""
        if event == "progress":
            task = args[0]
            QTimer.singleShot(0, lambda t=task: self.queue_tab.update_task(t))
        elif event == "status":
            task = args[0]
            QTimer.singleShot(0, lambda t=task: self.queue_tab.update_task(t))
        elif event == "log":
            task_id, msg = args[0], args[1]
            QTimer.singleShot(0, lambda tid=task_id, m=msg: self.log_tab.append(tid, m))
        elif event == "finished":
            task = args[0]
            QTimer.singleShot(0, lambda t=task: self._on_task_finished(t))
        elif event == "all_done":
            QTimer.singleShot(0, self._on_all_done)

    def _on_task_finished(self, task: DownloadTask):
        self.queue_tab.update_task(task)
        if task.status == DownloadStatus.COMPLETE.value:
            self.status_bar.showMessage(f"✅ Completed: {task.filename}")
            if self.settings.get("notifications", True) and hasattr(self, "tray"):
                self.tray.showMessage(
                    "Download Complete",
                    f"{task.filename} downloaded successfully!",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000
                )
        else:
            self.status_bar.showMessage(f"❌ Failed: {task.filename}")

    def _on_all_done(self):
        self.status_bar.showMessage("All downloads complete!")
        if self.settings.get("auto_shutdown", False):
            QMessageBox.information(self, "All Done", "All downloads complete. Shutting down in 30 seconds...")
            if platform.system() == "Windows":
                subprocess.run(["shutdown", "/s", "/t", "30"])
            else:
                subprocess.run(["shutdown", "-h", "+1"])

    def _update_status(self):
        active = self.queue_mgr.active_count()
        if active > 0:
            # Compute total speed
            total_speed = sum(
                t.speed for t in self.queue_mgr.tasks
                if t.status == DownloadStatus.ACTIVE.value
            )
            speed_str = f"{fmt_bytes(int(total_speed))}/s"
            self.hdr_status.setText(f"● {active} active  ↓ {speed_str}")
            self.hdr_status.setStyleSheet("color:#58a6ff; font-size:13px;")
        else:
            self.hdr_status.setText("● Idle")
            self.hdr_status.setStyleSheet("color:#3fb950; font-size:13px;")

    def _on_token_changed(self, token: str):
        if token:
            self.status_bar.showMessage("✅ HuggingFace token saved.")
        else:
            self.status_bar.showMessage("Logged out of HuggingFace.")

    def _on_clipboard_url(self, url: str):
        reply = QMessageBox.question(
            self, "URL Detected",
            f"HuggingFace URL detected in clipboard:\n\n{url[:80]}...\n\nAdd to download queue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            fname = Path(urllib.parse.urlparse(url).path).name or "download"
            task_data = {
                "id":       make_id(),
                "url":      url,
                "filename": fname,
                "dest_dir": self.settings.get("download_dir"),
                "method":   DownloadMethod.AUTO.value,
                "connections": self.settings.get("connections", 16),
                "max_retries": self.settings.get("max_retries", 10),
            }
            self._add_download(task_data)

    def _open_download_folder(self):
        folder = self.settings.get("download_dir")
        Path(folder).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {APP_VERSION}</p>"
            "<p>A production-quality downloader for HuggingFace models, "
            "especially large GGUF files.</p>"
            "<p>Features:<br>"
            "• aria2c multi-connection downloads<br>"
            "• Full resume support<br>"
            "• HuggingFace Hub integration<br>"
            "• GGUF quantization guide<br>"
            "• Download queue management<br>"
            "• Auto-retry on failure</p>"
            "<p>Log file: <code>" + str(LOG_FILE) + "</code></p>"
        )

    def closeEvent(self, event):
        if self.settings.get("tray_minimize") and hasattr(self, "tray") and self.tray.isVisible():
            event.ignore()
            self.hide()
            self.tray.showMessage(
                APP_NAME, "Running in background. Double-click tray icon to restore.",
                QSystemTrayIcon.MessageIcon.Information, 2000
            )
        else:
            # Stop clipboard monitor
            if hasattr(self, "clipboard_mon"):
                self.clipboard_mon.stop()
            event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """Application entry point."""
    # Ensure app dir exists
    APP_DIR.mkdir(parents=True, exist_ok=True)

    # Check Qt is available
    if QT_BINDING is None:
        print(
            "ERROR: No Qt binding found!\n"
            "Please install PySide6 or PyQt6:\n"
            "  pip install PySide6\n"
            "or\n"
            "  pip install PyQt6"
        )
        sys.exit(1)

    log.info(f"Using Qt binding: {QT_BINDING}")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("HFDownloadManager")

    # High DPI support
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Apply initial dark style
    app.setStyleSheet(DARK_STYLE)

    # Initialize core components early for wizard
    aria2 = Aria2Manager()

    # Run setup wizard if first launch or dependencies missing
    dep_status  = DependencyChecker.check_all()
    needs_setup = not all(dep_status.values()) or not aria2.available

    # Always show wizard on first run (no settings file)
    if not SETTINGS_FILE.exists() or needs_setup:
        wizard = SetupWizard(aria2)
        wizard.exec()

    # Launch main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
