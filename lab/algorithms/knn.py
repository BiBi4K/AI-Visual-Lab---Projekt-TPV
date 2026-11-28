"""Interactive 2D KNN playground. Run: python3 knn.py.

The UI uses Tkinter; plotting uses Matplotlib. The small, separate knn_core.py
contains the algorithm so you can study it without reading the interface code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

# Keep Matplotlib's optional cache out of the user's home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "knn-mpl-cache"))

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if __package__:
    from .knn_core import DatasetConfig, ModelConfig, KNNModel, generate_dataset, plot_limits, distance_outline, training_data
else:
    from knn_core import DatasetConfig, ModelConfig, KNNModel, generate_dataset, plot_limits, distance_outline, training_data


COLORS = ["#2878bf", "#e58027", "#209781", "#9361b5", "#d64e65", "#9d8035"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, len(COLORS) + 0.5), len(COLORS))


def draw_plot(ax, X, y, model, queries, limits, options, added=None, background=None):
    """Draw a view; shared by the desktop UI and the headless preview command."""
    ax.clear()
    ax.set_facecolor("#fbfcfe")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    if options["regions"]:
        if background is None:
            xx, yy = np.meshgrid(np.linspace(*limits[0], options["resolution"]),
                                 np.linspace(*limits[1], options["resolution"]))
            zz = model.predict(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
            background = (xx, yy, zz)
        xx, yy, zz = background
        ax.pcolormesh(xx, yy, zz, cmap=CMAP, norm=NORM, shading="nearest", alpha=0.16, zorder=0)
    for label in model.classes_:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=options["point_size"], c=COLORS[int(label)],
                   marker=MARKERS[int(label)], edgecolors="white", linewidths=0.8,
                   label=f"Class {int(label)}  ({mask.sum()})", zorder=3)
    if added is not None and added.any():
        ax.scatter(X[added, 0], X[added, 1], s=options["point_size"] + 50,
                   facecolors="none", edgecolors="#253247", linewidths=1, zorder=4)
    if queries:
        query_array = np.asarray(queries)
        predictions = model.predict(query_array)
        if len(queries) > 1:
            ax.scatter(query_array[:-1, 0], query_array[:-1, 1], marker="*", s=125,
                       c=[COLORS[int(i)] for i in predictions[:-1]], alpha=0.55,
                       edgecolors="#253247", linewidths=0.6, zorder=5)
        query = query_array[-1]
        indices, distances, _ = model.neighbors(query)
        if options["neighbors"]:
            for rank, idx in enumerate(indices, 1):
                neighbor = X[idx]
                ax.plot([query[0], neighbor[0]], [query[1], neighbor[1]],
                        color="#4e5d73", alpha=0.65, linewidth=1, linestyle="--", zorder=2)
                if options["ranks"]:
                    ax.annotate(str(rank), neighbor, xytext=(5, 5), textcoords="offset points",
                                fontsize=8, color="#1b293c", zorder=6,
                                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
            ax.scatter(X[indices, 0], X[indices, 1], s=options["point_size"] + 85,
                       facecolors="none", edgecolors="#253247", linewidths=1.4, zorder=4)
        if options["radius"] and distances[-1] > 0:
            outline = distance_outline(model, query, distances[-1])
            ax.plot(outline[:, 0], outline[:, 1], color="#53627a", linestyle=":", linewidth=1.5, zorder=2)
        ax.scatter(*query, s=330, marker="*", c=COLORS[int(predictions[-1])],
                   edgecolors="#16253a", linewidths=1.4, zorder=8)
    ax.set_xlabel("Feature 1", color="#53627a", labelpad=8)
    ax.set_ylabel("Feature 2", color="#53627a", labelpad=8)
    ax.tick_params(colors="#64748b", labelsize=9)
    if options["grid"]:
        ax.grid(True, alpha=0.12, color="#64748b")
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#d9e1eb")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", color="none", markerfacecolor="white",
                          markeredgecolor="#253247", markersize=12))
    labels.append("Query point")
    ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.95, edgecolor="#e2e8f0")
    return background


DEFAULT_VIEW = dict(regions=True, neighbors=True, radius=True, ranks=False,
                    grid=True, resolution=100, point_size=48)


class KNNApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("KNN Lab — interactive classifier")
        root.geometry("1320x900")
        root.minsize(1080, 800)
        root.configure(bg="#f3f6fa")
        self._pending = None
        self._ready = False
        self._background = None
        self._cache_key = None
        self._revision = 0
        self.queries = []
        self.history = []
        self.model = None
        self._style()

        header = ttk.Frame(root, padding=(22, 16, 22, 12))
        header.pack(fill="x")
        ttk.Label(header, text="KNN Lab", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Generate · explore · classify", style="Muted.TLabel").pack(side="left", padx=20)
        self.summary_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.summary_var, style="Muted.TLabel").pack(side="right")

        body = ttk.Frame(root, padding=(16, 0, 16, 0))
        body.pack(fill="both", expand=True)
        sidebar_container = ttk.Frame(body, width=314)
        sidebar_container.pack(side="left", fill="y", padx=(0, 16))
        sidebar_container.pack_propagate(False)
        sidebar_canvas = tk.Canvas(sidebar_container, width=296, highlightthickness=0, bg="#f3f6fa")
        sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_scroll.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar = ttk.Frame(sidebar_canvas)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        sidebar_canvas.bind("<Configure>", lambda event: sidebar_canvas.itemconfigure(sidebar_window, width=event.width))
        notebook = ttk.Notebook(sidebar)
        notebook.pack(fill="x")
        data_tab, model_tab, view_tab = [ttk.Frame(notebook, padding=14) for _ in range(3)]
        for tab, label in [(data_tab, "Dataset"), (model_tab, "Model"), (view_tab, "Display")]:
            notebook.add(tab, text=label)

        self.data_vars = {key: tk.StringVar(value=str(value)) for key, value in dict(
            dataset="Blobs", n_samples=180, n_classes=3, noise=0.65, separation=3,
            imbalance=1, seed=42, stretch=1, rotation=0).items()}
        self.data_widgets = {}
        self.data_widgets["dataset"] = self._combo(data_tab, "Pattern", self.data_vars["dataset"],
                                                   ["Blobs", "Moons", "Circles", "Spirals", "XOR"])
        for key, label, low, high, step in [
            ("n_samples", "Training points", 20, 1200, 20),
            ("n_classes", "Classes", 2, 6, 1),
            ("noise", "Noise / spread", 0, 2, 0.05),
            ("separation", "Pattern size", 0.2, 8, 0.2),
            ("imbalance", "Last-class ratio", 0.1, 1, 0.1),
            ("seed", "Random seed", 0, 999999, 1),
            ("stretch", "Feature 1 stretch", 0.1, 10, 0.1),
            ("rotation", "Rotation (degrees)", 0, 180, 5),
        ]:
            self.data_widgets[key] = self._spin(data_tab, label, self.data_vars[key], low, high, step)
        self.dataset_hint = tk.StringVar()
        ttk.Label(data_tab, textvariable=self.dataset_hint, style="Muted.TLabel", wraplength=244).pack(anchor="w", pady=(8, 10))
        ttk.Button(data_tab, text="Generate dataset", style="Accent.TButton", command=self.generate).pack(fill="x")
        ttk.Button(data_tab, text="New random seed + generate", command=self.randomize).pack(fill="x", pady=(6, 0))
        ttk.Label(data_tab, text="Generation clears edits and query points.", style="Small.TLabel", wraplength=244).pack(anchor="w", pady=(9, 0))
        self.data_vars["dataset"].trace_add("write", self._dataset_changed)

        self.model_vars = {key: tk.StringVar(value=str(value)) for key, value in dict(
            k=5, metric="euclidean", p=2, weights="uniform").items()}
        self.model_vars["standardize"] = tk.BooleanVar(value=False)
        self._spin(model_tab, "Neighbors (k)", self.model_vars["k"], 1, 100, 1)
        self._combo(model_tab, "Distance", self.model_vars["metric"], ["euclidean", "manhattan", "chebyshev", "minkowski"])
        self.p_widget = self._spin(model_tab, "Minkowski power (p)", self.model_vars["p"], 1, 8, 0.25)
        self._combo(model_tab, "Voting", self.model_vars["weights"], ["uniform", "distance"])
        ttk.Checkbutton(model_tab, text="Standardize features", variable=self.model_vars["standardize"]).pack(anchor="w", pady=(12, 10))
        ttk.Label(model_tab, text="Changes apply automatically.\n\nUniform: each neighbor has one vote.\nDistance: closer neighbors count more.\n\nStandardization balances feature scales. Try stretching Feature 1 first.\n\nClass-vote ties choose the lowest class number.",
                  style="Muted.TLabel", wraplength=242, justify="left").pack(anchor="w", pady=10)
        for var in self.model_vars.values():
            var.trace_add("write", self._model_changed)

        self.view_vars = {key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value)))
                          for key, value in DEFAULT_VIEW.items()}
        for key, label in [("regions", "Decision regions"), ("neighbors", "Highlight nearest neighbors"),
                           ("radius", "Distance to kth neighbor"), ("ranks", "Neighbor rank labels"), ("grid", "Plot grid")]:
            ttk.Checkbutton(view_tab, text=label, variable=self.view_vars[key]).pack(anchor="w", pady=6)
        self._spin(view_tab, "Region resolution", self.view_vars["resolution"], 40, 200, 20)
        self._spin(view_tab, "Point size", self.view_vars["point_size"], 15, 120, 5)
        ttk.Label(view_tab, text="Higher resolution makes the regions smoother but takes longer.\n\nThe dotted outline uses your chosen distance metric.\n\nStars are query points. Outlined training points were added by you.",
                  style="Muted.TLabel", wraplength=244, justify="left").pack(anchor="w", pady=12)
        for var in self.view_vars.values():
            var.trace_add("write", self._view_changed)

        interact = ttk.LabelFrame(sidebar, text="Click on the plot", padding=12)
        interact.pack(fill="x", pady=(14, 0))
        self.mode_var = tk.StringVar(value="Predict")
        for mode, label in [("Predict", "Predict a new point"), ("Add", "Add a labeled training point"), ("Remove", "Remove a training point")]:
            ttk.Radiobutton(interact, text=label, variable=self.mode_var, value=mode, command=self._mode_changed).pack(anchor="w", pady=3)
        self.class_var = tk.StringVar(value="0")
        self.class_combo = self._combo(interact, "Label for added point", self.class_var, ["0", "1", "2"])
        actions = ttk.Frame(interact)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Undo edit", command=self.undo).pack(side="left", fill="x", expand=True)
        ttk.Button(actions, text="Clear queries", command=self.clear_queries).pack(side="left", fill="x", expand=True, padx=(5, 0))
        ttk.Button(sidebar, text="Reset plot view", command=self.reset_view).pack(fill="x", pady=(12, 5))
        ttk.Button(sidebar, text="Export training data (.json)", command=self.export_json).pack(fill="x")

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.instruction_var = tk.StringVar()
        ttk.Label(right, textvariable=self.instruction_var, style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor="#ffffff")
        self.figure.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.97)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.pack(fill="x")
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.canvas.mpl_connect("button_release_event", self._plot_release)
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)

        inspector = ttk.LabelFrame(right, text="How this prediction is made", padding=(12, 8))
        inspector.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        details = ttk.Frame(inspector)
        details.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.prediction_var = tk.StringVar(value="Click the plot to inspect a prediction.")
        ttk.Label(details, textvariable=self.prediction_var, style="Heading.TLabel", wraplength=400).pack(anchor="w")
        self.votes_var = tk.StringVar(value="1. Measure distances\n2. Select the k closest training points\n3. Combine their class votes")
        ttk.Label(details, textvariable=self.votes_var, wraplength=400, justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(details, text="Vote share describes the neighbors; it is not measured accuracy.",
                  style="Small.TLabel", wraplength=400).pack(anchor="w", pady=(8, 0))
        table_frame = ttk.Frame(inspector)
        table_frame.pack(side="right", fill="y")
        self.neighbor_table = ttk.Treeview(table_frame, columns=("rank", "class", "distance", "vote"),
                                           show="headings", height=5, selectmode="none")
        for key, label, width in [("rank", "#", 32), ("class", "Class", 50), ("distance", "Distance", 80), ("vote", "Vote", 65)]:
            self.neighbor_table.heading(key, text=label)
            self.neighbor_table.column(key, width=width, minwidth=width, anchor="center", stretch=False)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.neighbor_table.yview)
        self.neighbor_table.configure(yscrollcommand=scroll.set)
        self.neighbor_table.pack(side="left")
        scroll.pack(side="left", fill="y")

        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var, style="Muted.TLabel", padding=(20, 9)).pack(fill="x")
        self._dataset_changed()
        self._mode_changed()
        self._ready = True
        self.generate()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Helvetica", 11), background="#f3f6fa", foreground="#253247")
        style.configure("TFrame", background="#f3f6fa")
        style.configure("TLabel", background="#f3f6fa")
        style.configure("Title.TLabel", font=("Helvetica", 25, "bold"), foreground="#16304f")
        style.configure("Heading.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("Muted.TLabel", foreground="#637189", font=("Helvetica", 10))
        style.configure("Small.TLabel", foreground="#637189", font=("Helvetica", 9))
        style.configure("TButton", padding=(8, 5))
        style.configure("Accent.TButton", background="#2476bd", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1a609e")])
        style.configure("TNotebook.Tab", padding=(12, 7))
        style.configure("Treeview", rowheight=22, background="white", fieldbackground="white")

    def _spin(self, parent, label, variable, low, high, step):
        row = self.ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        self.ttk.Label(row, text=label).pack(side="left")
        widget = self.ttk.Spinbox(row, textvariable=variable, from_=low, to=high, increment=step, width=8)
        widget.pack(side="right")
        return widget

    def _combo(self, parent, label, variable, choices):
        self.ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 3))
        widget = self.ttk.Combobox(parent, textvariable=variable, values=choices, state="readonly", width=22)
        widget.pack(fill="x", pady=(0, 4))
        return widget

    @staticmethod
    def _number(variables, key, minimum, maximum, integer=False):
        text = variables[key].get()
        number = float(text)
        if not np.isfinite(number) or not minimum <= number <= maximum or (integer and number != int(number)):
            raise ValueError(f"{key.replace('_', ' ')} must be {'an integer ' if integer else ''}between {minimum} and {maximum}.")
        return int(number) if integer else number

    def _dataset_changed(self, *_):
        binary = self.data_vars["dataset"].get() in {"Moons", "Circles", "XOR"}
        if binary:
            self.data_vars["n_classes"].set("2")
        self.data_widgets["n_classes"].configure(state="disabled" if binary else "normal")
        self.dataset_hint.set(("This pattern has 2 classes. " if binary else "Choose 2–6 classes. ") +
                              "Ratio 1 = balanced; smaller values reduce the last class.")

    def _mode_changed(self):
        mode = self.mode_var.get()
        hints = {"Predict": "Click to predict a point and inspect its neighbors",
                 "Add": "Click to add a training point with the selected label",
                 "Remove": "Click near a training point to remove it"}
        if hasattr(self, "instruction_var"):
            self.instruction_var.set(hints[mode])
        self.class_combo.configure(state="readonly" if mode == "Add" else "disabled")

    def _model_changed(self, *_):
        self.p_widget.configure(state="normal" if self.model_vars["metric"].get() == "minkowski" else "disabled")
        self._schedule()

    def _view_changed(self, *_):
        self._schedule()

    def _schedule(self):
        if not self._ready:
            return
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(200, self.refresh)

    def _get_model_config(self):
        return ModelConfig(k=self._number(self.model_vars, "k", 1, 100, True),
                           metric=self.model_vars["metric"].get(),
                           p=self._number(self.model_vars, "p", 1, 8) if self.model_vars["metric"].get() == "minkowski" else 2,
                           weights=self.model_vars["weights"].get(),
                           standardize=self.model_vars["standardize"].get())

    def _get_view(self):
        options = {key: var.get() for key, var in self.view_vars.items()}
        options["resolution"] = self._number(self.view_vars, "resolution", 40, 200, True)
        options["point_size"] = self._number(self.view_vars, "point_size", 15, 120)
        return options

    def generate(self):
        try:
            dataset = self.data_vars["dataset"].get()
            config = DatasetConfig(dataset=dataset,
                                   n_samples=self._number(self.data_vars, "n_samples", 20, 1200, True),
                                   n_classes=2 if dataset in {"Moons", "Circles", "XOR"} else self._number(self.data_vars, "n_classes", 2, 6, True),
                                   noise=self._number(self.data_vars, "noise", 0, 2),
                                   separation=self._number(self.data_vars, "separation", 0.2, 8),
                                   imbalance=self._number(self.data_vars, "imbalance", 0.1, 1),
                                   seed=self._number(self.data_vars, "seed", 0, 999999, True))
            stretch = self._number(self.data_vars, "stretch", 0.1, 10)
            rotation = self._number(self.data_vars, "rotation", 0, 180)
            model_config, _ = self._get_model_config(), self._get_view()
            X, y = generate_dataset(config, stretch, rotation)
            model = KNNModel(model_config).fit(X, y)
        except (ValueError, self.tk.TclError) as exc:
            self.status_var.set(f"Check the settings: {exc}")
            return
        self.X, self.y, self.model = X, y, model
        self.added = np.zeros(len(y), dtype=bool)
        self.history.clear()
        self.queries.clear()
        self.active_dataset = config.dataset
        self.class_combo.configure(values=[str(i) for i in range(config.n_classes)])
        if int(self.class_var.get()) >= config.n_classes:
            self.class_var.set("0")
        self._revision += 1
        self._set_limits(plot_limits(X))
        self.toolbar.update()
        self.refresh()

    def randomize(self):
        self.data_vars["seed"].set(str(np.random.default_rng().integers(0, 1000000)))
        self.generate()

    def _set_limits(self, limits):
        self.ax.set_xlim(*limits[0])
        self.ax.set_ylim(*limits[1])

    def refresh(self):
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = None
        if self.model is None:
            return
        try:
            config, options = self._get_model_config(), self._get_view()
        except (ValueError, self.tk.TclError) as exc:
            self.status_var.set(f"Check the settings: {exc}")
            return
        limits = (self.ax.get_xlim(), self.ax.get_ylim())
        key = (self._revision, repr(config), limits, options["resolution"])
        if key != self._cache_key:
            self.model = KNNModel(config).fit(self.X, self.y)
            self._background = None
            self._cache_key = key
        self._background = draw_plot(self.ax, self.X, self.y, self.model, self.queries,
                                     limits, options, self.added, self._background)
        self._drawn_limits = limits
        self.canvas.draw_idle()
        self._update_inspector()
        self.summary_var.set(f"{self.active_dataset}  ·  {len(self.y)} points  ·  k = {self.model.effective_k}")
        note = f"k limited to {len(self.y)} available points. " if config.k > len(self.y) else ""
        self.status_var.set(note + "Model and display settings update automatically. Turn off toolbar Pan/Zoom before adding points.")
        self.p_widget.configure(state="normal" if config.metric == "minkowski" else "disabled")

    def _update_inspector(self):
        self.neighbor_table.delete(*self.neighbor_table.get_children())
        if not self.queries:
            self.prediction_var.set("Click the plot to inspect a prediction.")
            self.votes_var.set("1. Measure distances\n2. Select the k closest training points\n3. Combine their class votes")
            return
        point = self.queries[-1]
        prediction = int(self.model.predict([point])[0])
        probabilities = self.model.predict_proba([point])[0]
        self.prediction_var.set(f"({point[0]:.2f}, {point[1]:.2f})  →  Class {prediction}")
        shares = "    ".join(f"C{int(c)}: {v:.0%}" for c, v in zip(self.model.classes_, probabilities))
        unit = "standardized units" if self.model.config.standardize else "original units"
        self.votes_var.set(f"Vote shares: {shares}\n{self.model.effective_k} neighbors · {self.model.config.weights} voting\nDistances use {unit}.")
        indices, distances, weights = self.model.neighbors(point)
        for rank, (idx, distance, weight) in enumerate(zip(indices, distances, weights), 1):
            self.neighbor_table.insert("", "end", values=(rank, self.y[idx], f"{distance:.4f}", f"{weight:.1%}"))

    def on_plot_click(self, event):
        """Matplotlib gives a click's position directly in feature coordinates."""
        if event.inaxes != self.ax or event.button != 1 or self.toolbar.mode or self.model is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata])
        if not np.all(np.isfinite(point)):
            return
        mode = self.mode_var.get()
        if mode == "Predict":
            self.queries.append(point)
            self.queries = self.queries[-100:]
        elif mode == "Add":
            if len(self.y) >= 1500:
                self.status_var.set("The interactive limit is 1500 training points. Remove points or generate a smaller dataset.")
                return
            self._remember_edit()
            self.X = np.vstack((self.X, point))
            self.y = np.append(self.y, int(self.class_var.get()))
            self.added = np.append(self.added, True)
            self._revision += 1
        elif mode == "Remove":
            # Pick in screen pixels so selection stays intuitive after zooming.
            distances = np.linalg.norm(self.ax.transData.transform(self.X) - [event.x, event.y], axis=1)
            idx = int(np.argmin(distances))
            if distances[idx] > 14:
                self.status_var.set("Click closer to a training point to remove it.")
                return
            if len(self.y) <= 1:
                self.status_var.set("Keep at least one training point.")
                return
            self._remember_edit()
            self.X, self.y, self.added = (np.delete(array, idx, axis=0) for array in (self.X, self.y, self.added))
            self._revision += 1
        self.refresh()

    def _remember_edit(self):
        self.history.append((self.X.copy(), self.y.copy(), self.added.copy()))
        self.history = self.history[-100:]

    def undo(self):
        if not self.history:
            self.status_var.set("There are no training-point edits to undo.")
            return
        self.X, self.y, self.added = self.history.pop()
        self._revision += 1
        self.refresh()

    def clear_queries(self):
        self.queries.clear()
        self.refresh()

    def reset_view(self):
        if self.model is not None:
            self._set_limits(plot_limits(self.X))
            self.toolbar.update()
            self.refresh()

    def _plot_release(self, event):
        if self.toolbar.mode:
            self._schedule()

    def _on_canvas_draw(self, event):
        # Toolbar Home/Back/Forward also change the visible region.
        if self._ready and hasattr(self, "_drawn_limits"):
            limits = (self.ax.get_xlim(), self.ax.get_ylim())
            if limits != self._drawn_limits:
                self._schedule()

    def export_json(self):
        from tkinter import filedialog
        destination = filedialog.asksaveasfilename(parent=self.root, title="Export training data",
                                                   defaultextension=".json", initialfile="knn_training_data.json", filetypes=[("JSON", "*.json")])
        if not destination:
            return
        try:
            with open(destination, "w", encoding="utf-8") as stream:
                json.dump(training_data(self.X, self.y, self.added), stream, indent=2)
        except OSError as exc:
            self.status_var.set(f"Could not save: {exc}")
            return
        self.status_var.set(f"Saved {len(self.y)} training points to {destination}")

    def close(self):
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self.root.destroy()


def save_preview(path):
    """Render an example without opening a desktop window (also useful on servers)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    X, y = generate_dataset(DatasetConfig())
    model = KNNModel(ModelConfig(k=7)).fit(X, y)
    figure = Figure(figsize=(9, 7.5), dpi=160, facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    figure.subplots_adjust(left=0.09, right=0.96, bottom=0.10, top=0.87)
    figure.text(0.09, 0.95, "KNN Lab", fontsize=24, weight="bold", color="#16304f")
    figure.text(0.09, 0.91, "7 neighbors · uniform voting · Euclidean distance", fontsize=11, color="#637189")
    query = np.array([0.7, 0.35])
    draw_plot(ax, X, y, model, [query], plot_limits(X), {**DEFAULT_VIEW, "resolution": 180})
    figure.savefig(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", metavar="PNG", help="Save a static example without opening a window")
    args = parser.parse_args()
    if args.preview:
        save_preview(args.preview)
        print(f"Saved preview: {args.preview}")
        return
    try:
        import tkinter as tk
        root = tk.Tk()
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(f"A desktop Python installation with Tkinter is required: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Could not open a desktop window: {exc}\nUse --preview example.png for a static image.") from exc
    KNNApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
