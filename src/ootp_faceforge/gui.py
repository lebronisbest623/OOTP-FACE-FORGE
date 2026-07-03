"""Tkinter GUI for OOTP FaceForge."""
from __future__ import annotations

import contextlib
import io
import os
import queue
import subprocess
import threading
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .cli import DEFAULT_OUT, PROJECT_ROOT, build_player, expand_path_batch_items


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_PROFILE = "identity"
APP_ICON = Path(__file__).resolve().parent / "assets" / "icon.ico"


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

        self.photos_var = tk.StringVar(
            value=str(PROJECT_ROOT / "photos_in" / "park_yongtaek")
        )
        self.status_var = tk.StringVar(value="Ready")
        self.photo_count_var = tk.StringVar(value="")
        self.selected_paths: list[Path] = [
            PROJECT_ROOT / "photos_in" / "park_yongtaek"
        ]

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
            text="Choose Folder",
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

        self.output_button = ttk.Button(
            controls,
            text="Open Result",
            command=self._open_output,
            state="disabled",
        )
        self.output_button.grid(row=8, column=0, sticky="ew", pady=(18, 0))

        preview_panel = ttk.Frame(body, style="Panel.TFrame", padding=24)
        preview_panel.grid(row=0, column=1, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(1, weight=1)

        ttk.Label(preview_panel, text="Preview", style="H2.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.preview = tk.Label(
            preview_panel,
            text="No Preview",
            bg="#ffffff",
            fg="#86868b",
            font=("Segoe UI", 13),
            anchor="center",
        )
        self.preview.grid(row=1, column=0, sticky="nsew", pady=(16, 0))

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
        self.photos_var.set(f"{len(paths)} selected photos")
        self.photo_count_var.set(f"{len(paths)} build{'s' if len(paths) != 1 else ''} ready")

    def _append_log(self, text: str) -> None:
        if text:
            self.logs.append(text)

    def _set_busy(self, busy: bool, label: str = "Building...",
                  total: int | None = None) -> None:
        if busy:
            self.status_var.set(label)
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
            return
        self.progress.stop()
        self.progress.configure(mode="indeterminate", value=0)
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
        if len(items) == 1:
            self._set_busy(True)
            thread = threading.Thread(target=self._build_worker, args=(items[0],), daemon=True)
        else:
            self._set_busy(True, f"Building 0/{len(items)}", total=len(items))
            thread = threading.Thread(target=self._batch_worker, args=(items,), daemon=True)
        thread.start()

    def _choose_batch_photos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose player photos",
            initialdir=str(PROJECT_ROOT / "photos_in"),
            filetypes=(
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files", "*.*"),
            ),
        )
        if paths:
            self.selected_paths = [Path(path) for path in paths]
            self._refresh_photo_count()

    def _choose_batch_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Choose a player folder or a folder containing player folders",
            initialdir=str(PROJECT_ROOT / "photos_in"),
        )
        if path:
            self.selected_paths = [Path(path)]
            self._refresh_photo_count()

    def _build_worker(self, item: dict) -> None:
        args = Namespace(
            photos=item["photos"],
            name=item.get("name"),
            slug=item.get("slug"),
            out_dir=str(DEFAULT_OUT),
            profile=item.get("profile", DEFAULT_PROFILE),
            texture_photo=item.get("texture_photo"),
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
            eye_detail_strength=item.get("eye_detail_strength"),
            detail_min_cos=item.get("detail_min_cos"),
            debug_dir=item.get("debug_dir"),
            size=512,
            aa=2,
            flat_copy=False,
            overwrite_meta=True,
        )
        self._apply_default_detail_args(args)
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
            for idx, item in enumerate(items, 1):
                if not isinstance(item, dict):
                    raise ValueError(f"batch item {idx} must be an object")
                if not item.get("photos"):
                    raise ValueError(f"batch item {idx} missing photos")
                name = item.get("name") or Path(str(item["photos"])).name
                self.events.put(("status", f"Building {idx}/{total}: {name}"))
                args = Namespace(
                    photos=item["photos"],
                    name=item.get("name"),
                    slug=item.get("slug"),
                    out_dir=item.get("out_dir", str(DEFAULT_OUT)),
                    profile=item.get("profile", DEFAULT_PROFILE),
                    texture_photo=item.get("texture_photo"),
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
                    detail_jpeg_quality=item.get("detail_jpeg_quality"),
                    eye_detail_strength=item.get("eye_detail_strength"),
                    detail_min_cos=item.get("detail_min_cos"),
                    debug_dir=item.get("debug_dir"),
                    size=item.get("size", 512),
                    aa=item.get("aa", 2),
                    flat_copy=item.get("flat_copy", False),
                    overwrite_meta=item.get("overwrite_meta", True),
                )
                self._apply_default_detail_args(args)
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        manifests.append(build_player(args))
                    self.events.put(("progress", (idx, total, f"Built {idx}/{total}: {name}")))
                except SystemExit as exc:
                    failures.append((name, str(exc)))
                    self.events.put(("progress", (idx, total, f"Skipped {idx}/{total}: {name}")))
                except Exception as exc:
                    failures.append((name, str(exc)))
                    self.events.put(("progress", (idx, total, f"Skipped {idx}/{total}: {name}")))
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
                    self._show_preview(Path(manifest["outputs"]["preview"]))
                    self.last_output = Path(manifest["outputs"]["preview"]).parents[1]
                    self.status_var.set("Complete")
                    self._set_busy(False)
                    self.output_button.configure(state="normal")
                elif kind == "batch_done":
                    manifests, failures, stdout, stderr = payload  # type: ignore[misc]
                    self._append_log(stdout)
                    self._append_log(stderr)
                    if manifests:
                        last_preview = Path(manifests[-1]["outputs"]["preview"])
                        self._show_preview(last_preview)
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
                    done, total, label = payload  # type: ignore[misc]
                    self.progress.configure(value=done, maximum=total)
                    self.status_var.set(str(label))
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
        self.after(100, self._poll_events)

    def _show_preview(self, path: Path) -> None:
        img = Image.open(path).convert("RGB")
        img.thumbnail((520, 520), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(img)
        self.preview.configure(image=self.preview_image, text="")

    def _open_output(self) -> None:
        if not self.last_output:
            return
        if os.name == "nt":
            subprocess.Popen(["explorer", str(self.last_output)])
        else:
            subprocess.Popen(["open", str(self.last_output)])


def main() -> None:
    from .landmarks import close_landmarker

    app = FaceForgeApp()
    try:
        app.mainloop()
    finally:
        close_landmarker()


if __name__ == "__main__":
    main()
