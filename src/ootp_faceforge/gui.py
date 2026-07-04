"""Tkinter GUI for OOTP FaceForge."""
from __future__ import annotations

import contextlib
import ctypes
import io
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
import uuid
from argparse import Namespace
from ctypes import wintypes
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .cli import (
    DEFAULT_OUT,
    WORKSPACE_PHOTOS,
    build_player,
    default_jobs,
    expand_path_batch_items,
    run_batch,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_PROFILE = "identity"
APP_ICON = Path(__file__).resolve().parent / "assets" / "icon.ico"

_CLSCTX_INPROC_SERVER = 0x1
_COINIT_APARTMENTTHREADED = 0x2
_RPC_E_CHANGED_MODE = 0x80010106
_HRESULT_CANCELLED = 0x800704C7
_FOS_PICKFOLDERS = 0x20
_FOS_FORCEFILESYSTEM = 0x40
_FOS_ALLOWMULTISELECT = 0x200
_FOS_PATHMUSTEXIST = 0x800
_SIGDN_FILESYSPATH = 0x80058000


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, value: str) -> None:
        parsed = uuid.UUID(value)
        data4 = (ctypes.c_ubyte * 8).from_buffer_copy(parsed.bytes[8:])
        super().__init__(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            data4,
        )


_CLSID_FILE_OPEN_DIALOG = _GUID("dc1c5a9c-e88a-4dde-a5a1-60f82a20aef7")
_IID_FILE_OPEN_DIALOG = _GUID("d57c7288-d4ad-4768-be02-9d969532d960")
_IID_SHELL_ITEM = _GUID("43826d1e-e718-42ee-bc55-a1e261c37bfe")


def _hr_value(hr: int) -> int:
    return int(hr) & 0xFFFFFFFF


def _hr_failed(hr: int) -> bool:
    return ctypes.c_long(hr).value < 0


def _raise_if_failed(hr: int) -> None:
    if _hr_failed(hr):
        raise OSError(f"HRESULT 0x{_hr_value(hr):08X}")


def _com_method(
    ptr: ctypes.c_void_p,
    index: int,
    restype: object,
    *argtypes: object,
) -> object:
    vtable = ctypes.cast(
        ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _release_com(ptr: ctypes.c_void_p | None) -> None:
    if ptr and ptr.value:
        release = _com_method(ptr, 2, wintypes.ULONG)
        release(ptr)


def _shell_item_from_path(path: Path) -> ctypes.c_void_p:
    shell32 = ctypes.windll.shell32
    shell32.SHCreateItemFromParsingName.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    shell32.SHCreateItemFromParsingName.restype = ctypes.c_long

    item = ctypes.c_void_p()
    hr = shell32.SHCreateItemFromParsingName(
        str(path),
        None,
        ctypes.byref(_IID_SHELL_ITEM),
        ctypes.byref(item),
    )
    _raise_if_failed(hr)
    return item


def _choose_folders_native(
    parent: tk.Tk,
    title: str,
    initialdir: Path,
) -> list[Path] | None:
    if os.name != "nt":
        return None

    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None

    dialog = ctypes.c_void_p()
    results = ctypes.c_void_p()
    folder_item = ctypes.c_void_p()
    should_uninitialize = False
    try:
        hr = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
        if hr in (0, 1):
            should_uninitialize = True
        elif _hr_value(hr) != _RPC_E_CHANGED_MODE:
            _raise_if_failed(hr)

        hr = ole32.CoCreateInstance(
            ctypes.byref(_CLSID_FILE_OPEN_DIALOG),
            None,
            _CLSCTX_INPROC_SERVER,
            ctypes.byref(_IID_FILE_OPEN_DIALOG),
            ctypes.byref(dialog),
        )
        _raise_if_failed(hr)

        options = wintypes.DWORD()
        get_options = _com_method(
            dialog, 10, ctypes.c_long, ctypes.POINTER(wintypes.DWORD)
        )
        set_options = _com_method(dialog, 9, ctypes.c_long, wintypes.DWORD)
        set_title = _com_method(dialog, 17, ctypes.c_long, wintypes.LPCWSTR)
        set_ok_label = _com_method(dialog, 18, ctypes.c_long, wintypes.LPCWSTR)
        set_default_folder = _com_method(
            dialog, 11, ctypes.c_long, ctypes.c_void_p
        )
        show = _com_method(dialog, 3, ctypes.c_long, wintypes.HWND)
        get_results = _com_method(
            dialog, 27, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
        )

        _raise_if_failed(get_options(dialog, ctypes.byref(options)))
        options.value |= (
            _FOS_PICKFOLDERS
            | _FOS_FORCEFILESYSTEM
            | _FOS_ALLOWMULTISELECT
            | _FOS_PATHMUSTEXIST
        )
        _raise_if_failed(set_options(dialog, options))
        _raise_if_failed(set_title(dialog, title))
        _raise_if_failed(set_ok_label(dialog, "Select Folders"))

        if initialdir.exists():
            folder_item = _shell_item_from_path(initialdir)
            _raise_if_failed(set_default_folder(dialog, folder_item))

        hr = show(dialog, parent.winfo_id())
        if _hr_value(hr) == _HRESULT_CANCELLED:
            return []
        _raise_if_failed(hr)
        _raise_if_failed(get_results(dialog, ctypes.byref(results)))

        get_count = _com_method(
            results, 7, ctypes.c_long, ctypes.POINTER(wintypes.DWORD)
        )
        get_item_at = _com_method(
            results,
            8,
            ctypes.c_long,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        )
        count = wintypes.DWORD()
        _raise_if_failed(get_count(results, ctypes.byref(count)))

        paths: list[Path] = []
        for index in range(count.value):
            item = ctypes.c_void_p()
            try:
                _raise_if_failed(get_item_at(results, index, ctypes.byref(item)))
                get_display_name = _com_method(
                    item,
                    5,
                    ctypes.c_long,
                    wintypes.DWORD,
                    ctypes.POINTER(ctypes.c_wchar_p),
                )
                raw_path = ctypes.c_wchar_p()
                _raise_if_failed(
                    get_display_name(item, _SIGDN_FILESYSPATH, ctypes.byref(raw_path))
                )
                try:
                    if raw_path.value:
                        paths.append(Path(raw_path.value))
                finally:
                    ole32.CoTaskMemFree(raw_path)
            finally:
                _release_com(item)
        return paths
    except Exception:
        return None
    finally:
        _release_com(results)
        _release_com(folder_item)
        _release_com(dialog)
        if should_uninitialize:
            ole32.CoUninitialize()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class FaceForgeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("OOTP FaceForge")
        self.geometry("1000x660")
        self.minsize(940, 600)
        self.configure(bg="#f5f5f7")
        self.window_icon: ImageTk.PhotoImage | None = None
        self._set_window_icon()

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_image: ImageTk.PhotoImage | None = None
        self.last_output: Path | None = None
        self.logs: list[str] = []

        default_photos = WORKSPACE_PHOTOS / "park_yongtaek"
        self.photos_var = tk.StringVar(value=str(default_photos))
        self.status_var = tk.StringVar(value="Ready")
        self.photo_count_var = tk.StringVar(value="")
        self.progress_detail_var = tk.StringVar(value="")
        self.preview_status_var = tk.StringVar(value="No completed preview yet")
        self.selected_paths: list[Path] = [default_photos]
        self._busy_started_at: float | None = None
        self._current_started_at: float | None = None
        self._progress_done = 0
        self._progress_total: int | None = None
        self._progress_failed = 0
        self._progress_workers: int | None = None
        self._progress_current = ""
        self._progress_current_index = 0
        self._progress_phase = ""
        self._last_progress_second = -1

        self._configure_style()
        self._build_ui()
        self._refresh_photo_count()
        self.after(100, self._poll_events)

    def _set_window_icon(self) -> None:
        if not APP_ICON.exists():
            return
        with contextlib.suppress(tk.TclError):
            self.iconbitmap(default=str(APP_ICON))
        with contextlib.suppress(Exception):
            icon = Image.open(APP_ICON).resize((64, 64), Image.Resampling.LANCZOS)
            self.window_icon = ImageTk.PhotoImage(icon)
            self.iconphoto(True, self.window_icon)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#f5f5f7")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("Subtle.TLabel", background="#f5f5f7", foreground="#6e6e73")
        style.configure("Panel.TLabel", background="#ffffff")
        style.configure("Title.TLabel", background="#f5f5f7", font=("Segoe UI", 28, "bold"))
        style.configure("H2.TLabel", background="#ffffff", font=("Segoe UI", 15, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#6e6e73")
        style.configure("Status.TLabel", background="#ffffff", foreground="#6e6e73")
        style.configure("TEntry", padding=8)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="App.TFrame", padding=(36, 28, 36, 10))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="OOTP FaceForge", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        body = ttk.Frame(self, style="App.TFrame", padding=(36, 18, 36, 32))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body, style="Panel.TFrame", padding=24)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Input", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        self.photo_button = tk.Button(
            controls,
            text="Choose Photos",
            command=self._choose_batch_photos,
            bg="#ffffff",
            fg="#0071e3",
            activebackground="#f5f5f7",
            activeforeground="#0071e3",
            bd=1,
            relief="solid",
            highlightthickness=0,
            font=("Segoe UI", 11, "bold"),
            padx=18,
            pady=12,
        )
        self.photo_button.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        self.folder_button = tk.Button(
            controls,
            text="Choose Folders",
            command=self._choose_batch_folder,
            bg="#ffffff",
            fg="#0071e3",
            activebackground="#f5f5f7",
            activeforeground="#0071e3",
            bd=1,
            relief="solid",
            highlightthickness=0,
            font=("Segoe UI", 11, "bold"),
            padx=18,
            pady=12,
        )
        self.folder_button.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.photo_path = ttk.Label(
            controls,
            textvariable=self.photos_var,
            style="Hint.TLabel",
            wraplength=300,
        )
        self.photo_path.grid(row=3, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.photo_count_var, style="Hint.TLabel").grid(
            row=4, column=0, sticky="w", pady=(4, 22)
        )

        self.build_button = tk.Button(
            controls,
            text="Build",
            command=self._build,
            bg="#0071e3",
            fg="#ffffff",
            activebackground="#147ce5",
            activeforeground="#ffffff",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 12, "bold"),
            padx=18,
            pady=13,
        )
        self.build_button.grid(row=5, column=0, sticky="ew")

        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=6, column=0, sticky="ew", pady=(18, 10))
        ttk.Label(controls, textvariable=self.status_var, style="Status.TLabel").grid(
            row=7, column=0, sticky="w"
        )
        ttk.Label(
            controls,
            textvariable=self.progress_detail_var,
            style="Hint.TLabel",
            wraplength=300,
        ).grid(row=8, column=0, sticky="w", pady=(4, 0))

        self.output_button = ttk.Button(
            controls,
            text="Open Result",
            command=self._open_output,
            state="disabled",
        )
        self.output_button.grid(row=9, column=0, sticky="ew", pady=(18, 0))

        preview_panel = ttk.Frame(body, style="Panel.TFrame", padding=24)
        preview_panel.grid(row=0, column=1, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(2, weight=1)

        ttk.Label(preview_panel, text="Preview", style="H2.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            preview_panel,
            textvariable=self.preview_status_var,
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.preview = tk.Label(
            preview_panel,
            text="No Preview",
            bg="#ffffff",
            fg="#86868b",
            font=("Segoe UI", 13),
            anchor="center",
        )
        self.preview.grid(row=2, column=0, sticky="nsew", pady=(16, 0))

    def _refresh_photo_count(self) -> None:
        paths = self.selected_paths
        if not paths:
            self.photos_var.set("")
            self.photo_count_var.set("Folder not found")
            return
        if len(paths) == 1 and paths[0].is_dir():
            photos = paths[0]
            self.photos_var.set(str(photos))
            count = sum(1 for p in photos.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if count:
                self.photo_count_var.set(f"{count} photo{'s' if count != 1 else ''} ready")
                return
            with contextlib.suppress(ValueError):
                items = expand_path_batch_items(paths)
                self.photo_count_var.set(f"{len(items)} folder{'s' if len(items) != 1 else ''} ready")
                return
            self.photo_count_var.set("No photos found")
            return
        dirs = sum(1 for p in paths if p.is_dir())
        files = sum(1 for p in paths if p.is_file())
        if dirs and not files:
            self.photos_var.set(f"{dirs} selected folders")
        elif files and not dirs:
            self.photos_var.set(f"{files} selected photos")
        else:
            self.photos_var.set(f"{len(paths)} selected items")
        with contextlib.suppress(ValueError):
            items = expand_path_batch_items(paths)
            self.photo_count_var.set(f"{len(items)} build{'s' if len(items) != 1 else ''} ready")
            return
        self.photo_count_var.set(f"{len(paths)} selection{'s' if len(paths) != 1 else ''} ready")

    def _append_log(self, text: str) -> None:
        if text:
            self.logs.append(text)

    def _short_name(self, value: str, limit: int = 34) -> str:
        value = str(value)
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "..."

    def _eta_seconds(self) -> float | None:
        total = self._progress_total
        done = self._progress_done
        if (
            self._busy_started_at is None
            or total is None
            or total <= 0
            or done <= 0
            or done >= total
        ):
            return None
        elapsed = time.perf_counter() - self._busy_started_at
        return (elapsed / done) * (total - done)

    def _update_progress_detail(self) -> None:
        if self._busy_started_at is None:
            return
        now = time.perf_counter()
        elapsed = _format_duration(now - self._busy_started_at)
        total = self._progress_total
        if total:
            done = min(self._progress_done, total)
            percent = int(round((done / max(total, 1)) * 100))
            pieces = [f"{done}/{total} done", f"{percent}%"]
            if self._progress_failed:
                pieces.append(f"{self._progress_failed} skipped")
            if self._progress_workers and self._progress_workers > 1:
                workers = min(self._progress_workers, total)
                pieces.append(f"{workers} workers")
            pieces.append(f"elapsed {elapsed}")
            eta = self._eta_seconds()
            if done >= total:
                pieces.append("ETA 00:00")
            else:
                pieces.append(f"ETA {_format_duration(eta) if eta is not None else '--:--'}")
            if (
                self._progress_phase == "building"
                and self._progress_current
                and self._progress_current_index > 0
            ):
                current_elapsed = _format_duration(
                    now - (self._current_started_at or self._busy_started_at)
                )
                pieces.append(
                    f"current {self._progress_current_index}/{total} "
                    f"{self._short_name(self._progress_current)} "
                    f"({current_elapsed})"
                )
            self.progress_detail_var.set(" • ".join(pieces))
            return
        self.progress_detail_var.set(f"elapsed {elapsed}")

    def _refresh_progress_clock(self) -> None:
        if self._busy_started_at is None:
            return
        second = int(time.perf_counter())
        if second == self._last_progress_second:
            return
        self._last_progress_second = second
        self._update_progress_detail()

    def _set_busy(self, busy: bool, label: str = "Building...",
                  total: int | None = None,
                  workers: int | None = None) -> None:
        if busy:
            self._busy_started_at = time.perf_counter()
            self._current_started_at = self._busy_started_at
            self._progress_done = 0
            self._progress_total = total
            self._progress_failed = 0
            self._progress_workers = workers
            self._progress_current = ""
            self._progress_current_index = 0
            self._progress_phase = "building"
            self._last_progress_second = -1
            self.status_var.set(label)
            self.progress_detail_var.set("")
            self.output_button.configure(state="disabled")
            self.build_button.configure(state="disabled")
            self.photo_button.configure(state="disabled")
            self.folder_button.configure(state="disabled")
            self.progress.stop()
            if total:
                self.progress.configure(mode="determinate", maximum=total, value=0)
            else:
                self.progress.configure(mode="indeterminate", value=0)
                self.progress.start(12)
            self._update_progress_detail()
            return
        self.progress.stop()
        if self._progress_total:
            self.progress.configure(
                mode="determinate",
                maximum=self._progress_total,
                value=min(self._progress_done, self._progress_total),
            )
        else:
            self.progress.configure(mode="indeterminate", value=0)
        self._busy_started_at = None
        self._current_started_at = None
        self._progress_total = None
        self._progress_workers = None
        self._progress_phase = ""
        self.build_button.configure(state="normal", text="Build")
        self.photo_button.configure(state="normal")
        self.folder_button.configure(state="normal")

    def _build(self) -> None:
        try:
            items = expand_path_batch_items(self.selected_paths)
        except ValueError as exc:
            messagebox.showerror("No photos", str(exc))
            return
        self.logs.clear()
        self.build_button.configure(text="Building...")
        self.preview_status_var.set("Preview updates after each completed build")
        if len(items) == 1:
            name = items[0].get("name") or Path(str(items[0]["photos"])).name
            self._set_busy(True, f"Building: {name}", workers=1)
            self._progress_current = name
            self._progress_current_index = 1
            self._progress_phase = "building"
            self._update_progress_detail()
            thread = threading.Thread(target=self._build_worker, args=(items[0],), daemon=True)
        else:
            jobs = min(default_jobs(), len(items))
            active_workers = jobs if jobs > 1 and len(items) > 2 else 1
            worker_label = "worker" if active_workers == 1 else "workers"
            self._set_busy(
                True,
                f"Building 0/{len(items)} with {active_workers} {worker_label}",
                total=len(items),
                workers=active_workers,
            )
            thread = threading.Thread(target=self._batch_worker, args=(items,), daemon=True)
        thread.start()

    def _choose_batch_photos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose player photos",
            initialdir=str(WORKSPACE_PHOTOS),
            filetypes=(
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files", "*.*"),
            ),
        )
        if paths:
            self.selected_paths = [Path(path) for path in paths]
            self._refresh_photo_count()

    def _choose_batch_folder(self) -> None:
        paths = _choose_folders_native(
            self,
            "Choose player folders",
            WORKSPACE_PHOTOS,
        )
        if paths is None:
            path = filedialog.askdirectory(
                title="Choose a player folder or a folder containing player folders",
                initialdir=str(WORKSPACE_PHOTOS),
            )
            paths = [Path(path)] if path else []
        if paths:
            self.selected_paths = paths
            self._refresh_photo_count()

    def _namespace_for_item(self, item: dict) -> Namespace:
        args = Namespace(
            photos=item["photos"],
            name=item.get("name"),
            slug=item.get("slug"),
            out_dir=item.get("out_dir", str(DEFAULT_OUT)),
            profile=item.get("profile", DEFAULT_PROFILE),
            texture_photo=item.get("texture_photo"),
            texture_mode=item.get("texture_mode", "fuse"),
            max_yaw=item.get("max_yaw"),
            shape_lam=item.get("shape_lam"),
            asym_lam=item.get("asym_lam"),
            dense_weight=item.get("dense_weight"),
            tex_lam=item.get("tex_lam"),
            tex_erode=item.get("tex_erode"),
            exposure_lo=item.get("exposure_lo"),
            exposure_hi=item.get("exposure_hi"),
            detail_size=item.get("detail_size"),
            detail_strength=item.get("detail_strength"),
            detail_chroma_strength=item.get("detail_chroma_strength"),
            detail_edge_strength=item.get("detail_edge_strength"),
            detail_flat_neutralize=item.get("detail_flat_neutralize"),
            detail_shadow_neutralize=item.get("detail_shadow_neutralize"),
            detail_jpeg_quality=item.get("detail_jpeg_quality"),
            eye_detail_strength=item.get("eye_detail_strength"),
            detail_min_cos=item.get("detail_min_cos"),
            restore=item.get("restore"),
            restore_model=item.get("restore_model"),
            id_refine=item.get("id_refine"),
            id_model=item.get("id_model"),
            refine_size=item.get("refine_size"),
            refine_r_max=item.get("refine_r_max"),
            debug_dir=item.get("debug_dir"),
            size=item.get("size", 512),
            aa=item.get("aa", 2),
            flat_copy=item.get("flat_copy", False),
            overwrite_meta=item.get("overwrite_meta", True),
        )
        self._apply_default_detail_args(args)
        return args

    def _build_worker(self, item: dict) -> None:
        args = self._namespace_for_item(item)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                manifest = build_player(args)
            self.events.put(("done", (manifest, stdout.getvalue(), stderr.getvalue())))
        except SystemExit as exc:
            self.events.put(("error", (exc, stdout.getvalue(), stderr.getvalue())))
        except Exception as exc:
            self.events.put(("error", (exc, stdout.getvalue(), stderr.getvalue())))

    def _batch_worker(self, items: list[dict]) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        manifests = []
        failures = []
        try:
            total = len(items)
            if total == 0:
                raise ValueError("batch is empty")
            started_at = time.perf_counter()
            namespaces: list[Namespace] = []
            for idx, item in enumerate(items, 1):
                if not isinstance(item, dict):
                    raise ValueError(f"batch item {idx} must be an object")
                if not item.get("photos"):
                    raise ValueError(f"batch item {idx} missing photos")
                namespaces.append(self._namespace_for_item(item))

            jobs = min(default_jobs(), total)
            active_workers = jobs if jobs > 1 and total > 2 else 1
            worker_label = "worker" if active_workers == 1 else "workers"
            self.events.put((
                "progress",
                {
                    "done": 0,
                    "total": total,
                    "failed": 0,
                    "current_index": 0,
                    "current": f"{active_workers} {worker_label} active",
                    "phase": "building",
                    "started_at": started_at,
                    "current_started_at": started_at,
                    "workers": active_workers,
                },
            ))

            failed_count = 0

            def completed(done: int, total: int, name: str,
                          err: str | None, manifest: dict | None) -> None:
                nonlocal failed_count
                if err:
                    failed_count += 1
                elif manifest is not None:
                    manifests.append(manifest)
                    self.events.put(("preview", manifest))
                self.events.put((
                    "progress",
                    {
                        "done": done,
                        "total": total,
                        "failed": failed_count,
                        "current_index": done,
                        "current": name,
                        "phase": "skipped" if err else "built",
                        "started_at": started_at,
                        "current_started_at": None,
                        "workers": active_workers,
                    },
                ))

            failures = run_batch(namespaces, jobs=jobs, completed=completed)
            self.events.put((
                "batch_done",
                (manifests, failures, stdout.getvalue(), stderr.getvalue()),
            ))
        except Exception as exc:
            self.events.put(("error", (exc, stdout.getvalue(), stderr.getvalue())))

    def _apply_default_detail_args(self, args: Namespace) -> None:
        for name in (
            "detail_chroma_strength",
            "detail_edge_strength",
            "detail_flat_neutralize",
            "detail_shadow_neutralize",
            "detail_jpeg_quality",
        ):
            if not hasattr(args, name):
                setattr(args, name, None)

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "done":
                    manifest, stdout, stderr = payload  # type: ignore[misc]
                    self._append_log(stdout)
                    self._append_log(stderr)
                    self._show_preview(
                        Path(manifest["outputs"]["preview"]),
                        f"Built: {manifest['player']['name']}",
                    )
                    self.last_output = Path(manifest["outputs"]["preview"]).parents[1]
                    self.status_var.set("Complete")
                    self._set_busy(False)
                    self.output_button.configure(state="normal")
                elif kind == "preview":
                    manifest = payload  # type: ignore[assignment]
                    preview_path = Path(manifest["outputs"]["preview"])
                    self._show_preview(preview_path, f"Built: {manifest['player']['name']}")
                    self.last_output = Path(manifest["outputs"]["fg"]).parents[2]
                    self.output_button.configure(state="normal")
                elif kind == "batch_done":
                    manifests, failures, stdout, stderr = payload  # type: ignore[misc]
                    self._append_log(stdout)
                    self._append_log(stderr)
                    if manifests:
                        last_preview = Path(manifests[-1]["outputs"]["preview"])
                        self._show_preview(
                            last_preview,
                            f"Last built: {manifests[-1]['player']['name']}",
                        )
                        self.last_output = Path(manifests[-1]["outputs"]["fg"]).parents[2]
                        self.output_button.configure(state="normal")
                    skipped = len(failures)
                    self.status_var.set(
                        f"Batch complete: {len(manifests)} built"
                        + (f", {skipped} skipped" if skipped else "")
                    )
                    self._set_busy(False)
                    if failures:
                        names = ", ".join(name for name, _ in failures[:4])
                        if len(failures) > 4:
                            names += f", +{len(failures) - 4} more"
                        messagebox.showwarning("Batch skipped photos", names)
                elif kind == "progress":
                    if isinstance(payload, dict):
                        done = int(payload.get("done", 0))
                        total = int(payload.get("total", 0))
                        current = str(payload.get("current", ""))
                        phase = str(payload.get("phase", "building"))
                        current_index = int(payload.get("current_index", done))
                        self._busy_started_at = float(
                            payload.get("started_at") or self._busy_started_at
                            or time.perf_counter()
                        )
                        current_started = payload.get("current_started_at")
                        if current_started is not None:
                            self._current_started_at = float(current_started)
                        self._progress_done = done
                        self._progress_total = total
                        self._progress_failed = int(payload.get("failed", 0))
                        workers = payload.get("workers")
                        if workers is not None:
                            self._progress_workers = int(workers)
                        self._progress_current = current
                        self._progress_current_index = current_index
                        self._progress_phase = phase
                        self.progress.configure(value=done, maximum=total)
                        if phase == "building":
                            self.status_var.set(
                                f"Building {current_index}/{total}: {current}"
                            )
                        elif phase == "built":
                            self.status_var.set(
                                f"Built {done}/{total}: {current}"
                            )
                        elif phase == "skipped":
                            self.status_var.set(
                                f"Skipped {done}/{total}: {current}"
                            )
                        else:
                            self.status_var.set(str(current))
                        self._update_progress_detail()
                    else:
                        done, total, label = payload  # type: ignore[misc]
                        self._progress_done = int(done)
                        self._progress_total = int(total)
                        self.progress.configure(value=done, maximum=total)
                        self.status_var.set(str(label))
                        self._update_progress_detail()
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "error":
                    exc, stdout, stderr = payload  # type: ignore[misc]
                    self._append_log(stdout)
                    self._append_log(stderr)
                    self._append_log(str(exc))
                    self.status_var.set("Failed")
                    self._set_busy(False)
                    messagebox.showerror("Build failed", str(exc))
        except queue.Empty:
            pass
        self._refresh_progress_clock()
        self.after(100, self._poll_events)

    def _show_preview(self, path: Path, caption: str | None = None) -> None:
        img = Image.open(path).convert("RGB")
        img.thumbnail((520, 520), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(img)
        self.preview.configure(image=self.preview_image, text="")
        if caption:
            self.preview_status_var.set(caption)

    def _open_output(self) -> None:
        if not self.last_output:
            return
        if os.name == "nt":
            subprocess.Popen(["explorer", str(self.last_output)])
        else:
            subprocess.Popen(["open", str(self.last_output)])


def main() -> None:
    import multiprocessing

    from .landmarks import close_landmarker

    multiprocessing.freeze_support()
    app = FaceForgeApp()
    try:
        app.mainloop()
    finally:
        close_landmarker()


if __name__ == "__main__":
    main()
