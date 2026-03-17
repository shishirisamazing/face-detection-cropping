import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
import sys


def _import_app_module(event_queue):
    """Import the heavy application module in a background thread."""
    try:
        event_queue.put(("progress", 12, "Loading application modules..."))
        import main.__main__ as app_main
        event_queue.put(("progress", 50, "Application modules loaded"))
        event_queue.put(("ready", app_main))
    except Exception:
        event_queue.put(("error", traceback.format_exc()))


def _build_loading_screen():
    """Create and return loading screen widgets/state."""
    state = {}

    root = tk.Tk()
    root.title("FaceCrop")
    root.resizable(False, False)
    root.configure(bg="#1e1e1e")
    root.protocol("WM_DELETE_WINDOW", lambda: None)

    win_w, win_h = 430, 170
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    pos_x = (screen_w - win_w) // 2
    pos_y = (screen_h - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

    title_label = tk.Label(
        root,
        text="FaceCrop",
        font=("Segoe UI", 18, "bold"),
        fg="#ffffff",
        bg="#1e1e1e",
    )
    title_label.pack(pady=(18, 6))

    status_text = tk.StringVar(value="Starting...")
    status_label = tk.Label(
        root,
        textvariable=status_text,
        font=("Segoe UI", 10),
        fg="#c8c8c8",
        bg="#1e1e1e",
    )
    status_label.pack(pady=(0, 8))

    progress_value = tk.IntVar(value=0)
    progress = ttk.Progressbar(
        root,
        orient="horizontal",
        length=360,
        mode="determinate",
        maximum=100,
        variable=progress_value,
    )
    progress.pack(pady=(0, 6))

    percent_text = tk.StringVar(value="0%")
    percent_label = tk.Label(
        root,
        textvariable=percent_text,
        font=("Segoe UI", 10, "bold"),
        fg="#4da3ff",
        bg="#1e1e1e",
    )
    percent_label.pack()

    state["root"] = root
    state["status_text"] = status_text
    state["progress_value"] = progress_value
    state["percent_text"] = percent_text
    return state


def _set_loader_progress(loader, percent, message):
    """Set loader progress to an explicit step and refresh UI."""
    percent = max(0, min(100, int(percent)))
    loader["status_text"].set(str(message))
    loader["progress_value"].set(percent)
    loader["percent_text"].set("{}%".format(percent))
    loader["root"].update_idletasks()
    loader["root"].update()


def _show_loading_screen():
    """Display a startup loading UI and return (loader, app_module, error_text)."""
    event_queue = queue.Queue()
    loader = _build_loading_screen()
    state = {
        "module": None,
        "error": None,
    }

    _set_loader_progress(loader, 3, "Starting...")

    if sys.platform == "darwin":
        try:
            _set_loader_progress(loader, 12, "Loading application modules...")
            import main.__main__ as app_main
            _set_loader_progress(loader, 50, "Application modules loaded")
            return loader, app_main, None
        except Exception:
            return loader, None, traceback.format_exc()

    worker = threading.Thread(target=_import_app_module, args=(event_queue,), daemon=True)
    worker.start()

    while state["module"] is None and state["error"] is None:
        try:
            while True:
                event = event_queue.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _set_loader_progress(loader, event[1], event[2])
                elif kind == "ready":
                    state["module"] = event[1]
                elif kind == "error":
                    state["error"] = str(event[1])
        except queue.Empty:
            pass

        # Keep loader responsive while waiting for background import.
        try:
            loader["root"].update_idletasks()
            loader["root"].update()
        except tk.TclError:
            state["error"] = "Loader window was closed during startup."
            break

        time.sleep(0.02)

    return loader, state["module"], state["error"]


def _show_startup_error(error_text):
    """Show startup errors in a simple dialog and also print to console."""
    print(error_text)
    dialog_root = tk.Tk()
    dialog_root.withdraw()
    messagebox.showerror("FaceCrop Startup Error", error_text)
    dialog_root.destroy()

if __name__ == "__main__":
    loader, module, startup_error = _show_loading_screen()

    if startup_error is not None or module is None:
        try:
            loader["root"].destroy()
        except Exception:
            pass
        _show_startup_error(startup_error or "Unknown startup error")
        raise SystemExit(1)

    try:
        _set_loader_progress(loader, 60, "Initializing Qt runtime...")
        app = module.create_qt_app(sys.argv)

        _set_loader_progress(loader, 80, "Building user interface...")
        main_window, _ui = module.create_main_window('english')  # only supports french or english

        _set_loader_progress(loader, 95, "Finalizing startup...")

        _set_loader_progress(loader, 100, "Launching FaceCrop...")
        loader["root"].destroy()

        sys.exit(module.run_qt_app(app, main_window))
    except Exception:
        try:
            loader["root"].destroy()
        except Exception:
            pass
        _show_startup_error(traceback.format_exc())
        raise SystemExit(1)
