"""
plot_pso_convergence.py
-----------------------
Offline utility to parse a PSO log file and produce a publication-quality
convergence plot (gbest_y vs. iteration).

Usage
-----
    python plot_pso_convergence.py path/to/job.out              # saves next to the log
    python plot_pso_convergence.py path/to/job.out -o fig.png   # explicit output path
    python plot_pso_convergence.py path/to/job.out --show        # also opens an interactive window

Can also be imported and called directly:

    from plot_pso_convergence import plot_gbest_y_hist
    plot_gbest_y_hist("job_2519260.out", output_path="convergence.pdf")
"""

import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams


# ── Typography / rcParams for publication quality ───────────────────────────

def _apply_paper_style() -> None:
    """Apply rcParams suitable for IEEE / Elsevier / NeurIPS-style figures."""
    rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        # Lines & axes
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   4,
        "ytick.major.size":   4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size":   2,
        "ytick.minor.size":   2,
        # Grid
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.45,
        "grid.alpha":         0.5,
        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # PDF/PS metadata
        "pdf.fonttype":       42,   # embed TrueType in PDF
        "ps.fonttype":        42,
    })


# ── Log parser ───────────────────────────────────────────────────────────────

def parse_gbest_y_hist(log_path: str | Path) -> list[float]:
    """
    Parse a PSO log file and return the list of global-best fitness values,
    one per outer iteration, in iteration order.

    Searches for lines matching the pattern:
        "PSO iteration {N} completed. Global best fitness: {value}"
    and falls back to any line containing "Global best fitness: {value}".

    Parameters
    ----------
    log_path : str or Path
        Path to the log file (.out or .txt).

    Returns
    -------
    list[float]
        Ordered list of gbest_y values (one per outer PSO iteration found).

    Raises
    ------
    FileNotFoundError
        If the log file does not exist.
    ValueError
        If no matching lines were found.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Primary pattern: "PSO iteration N completed. Global best fitness: VAL"
    primary_re = re.compile(
        r"PSO\s+iteration\s+(\d+)\s+completed\.\s+Global\s+best\s+fitness\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    # Fallback: any line that contains "Global best fitness: VAL"
    fallback_re = re.compile(
        r"Global\s+best\s+fitness\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )

    records: dict[int, float] = {}   # iteration index → value (primary)
    fallback_values: list[float] = []

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = primary_re.search(line)
            if m:
                it_idx = int(m.group(1))
                val    = float(m.group(2))
                records[it_idx] = val
                continue
            m2 = fallback_re.search(line)
            if m2:
                fallback_values.append(float(m2.group(1)))

    if records:
        # Sort by iteration index to guarantee correct order
        gbest_y_hist = [records[k] for k in sorted(records)]
    elif fallback_values:
        gbest_y_hist = fallback_values
    else:
        raise ValueError(
            "No 'Global best fitness' entries found in the log file. "
            "Please verify the file format."
        )

    return gbest_y_hist


# ── Plotter ──────────────────────────────────────────────────────────────────

def plot_gbest_y_hist(
    log_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    negate: bool = False,
) -> Path:
    """
    Parse *log_path* and save a publication-quality PSO convergence figure.

    Parameters
    ----------
    log_path : str or Path
        Path to the PSO log file.
    output_path : str or Path, optional
        Destination for the saved figure.  Defaults to the same directory as
        the log file, with the stem ``<log_stem>_convergence.pdf``.
        Supported extensions: .pdf, .png, .svg, .eps.
    title : str, optional
        Custom figure title.  Defaults to "PSO Convergence".
    show : bool
        If True, call ``plt.show()`` after saving.
    negate : bool
        If True, negate every value before plotting (useful when the stored
        fitness is the negation of a maximization objective).

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    log_path = Path(log_path).resolve()

    # ── Parse ────────────────────────────────────────────────────────────────
    gbest_y_hist = parse_gbest_y_hist(log_path)
    n_iters = len(gbest_y_hist)

    if negate:
        gbest_y_hist = [-v for v in gbest_y_hist]

    # ── Resolve output path ───────────────────────────────────────────────────
    if output_path is None:
        output_path = log_path.parent / f"{log_path.stem}_convergence.pdf"
    output_path = Path(output_path).resolve()

    # ── Style ─────────────────────────────────────────────────────────────────
    _apply_paper_style()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.8))   # single-column width ~3.5in; double ~7.0in

    iters = list(range(n_iters))

    # Shaded region from initial value down to final
    ax.fill_between(
        iters, gbest_y_hist,
        y2=max(gbest_y_hist),
        alpha=0.12,
        color="#1a4e8c",
        linewidth=0,
    )

    # Main convergence line
    ax.plot(
        iters, gbest_y_hist,
        color="#1a4e8c",
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        markeredgecolor="#1a4e8c",
        zorder=3,
        label="Global best fitness",
    )

    # Annotate the final (best) value
    best_val = gbest_y_hist[-1]
    ax.annotate(
        f"{best_val:.4g}",
        xy=(iters[-1], best_val),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="#1a4e8c",
        arrowprops=dict(arrowstyle="-", color="#1a4e8c", lw=0.8),
    )

    # ── Axes labels & title ──────────────────────────────────────────────────
    ax.set_xlabel("Iteration", labelpad=6)
    ax.set_ylabel("Global Best Fitness", labelpad=6)
    ax.set_title(
        title if title else "PSO Convergence",
        pad=10,
        fontweight="bold",
    )

    # Integer x-ticks aligned to actual iterations
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=min(n_iters, 10)))
    ax.set_xlim(left=-0.3, right=n_iters - 0.7)

    # Optional: y-axis scientific notation when values are very small/large
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"[plot_gbest_y_hist] Figure saved → {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PSO log file and plot the gbest convergence curve."
    )
    parser.add_argument("log_path", help="Path to the PSO log file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output figure path (default: <log_stem>_convergence.pdf next to the log)",
    )
    parser.add_argument(
        "--title", default=None,
        help="Custom figure title",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open an interactive matplotlib window after saving",
    )
    parser.add_argument(
        "--negate", action="store_true",
        help="Negate fitness values before plotting (for maximization problems stored as negatives)",
    )
    args = parser.parse_args()

    plot_gbest_y_hist(
        log_path=args.log_path,
        output_path=args.output,
        title=args.title,
        show=args.show,
        negate=args.negate,
    )


if __name__ == "__main__":
    _cli()