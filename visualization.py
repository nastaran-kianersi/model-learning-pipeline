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

    iters = list(range(1, n_iters + 1))

    # Shaded region from initial value down to final
    # ax.fill_between(
    #     iters, gbest_y_hist,
    #     y2=max(gbest_y_hist),
    #     alpha=0.12,
    #     color="#1a4e8c",
    #     linewidth=0,
    # )

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
        f"{best_val:.2g}",
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
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=min(n_iters, 10)))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) 
    # show every 2 iterations starting from 1
    # setting the xticks labels manually using a list from 1 to n_iters with a step of 2
    ax.set_xticks(list(range(1, n_iters + 1, 2)))
    # ax.set_xlim(left=-0.3, right=n_iters - 0.7)
    ax.set_xlim(left=0.8, right=n_iters + 0.3)

    # setting the x axis ticks to be divided into 2 equal parts between the major ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # Optional: y-axis scientific notation when values are very small/large
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "eps" if str(output_path).endswith(".eps") else None
    fig.savefig(output_path, format=fmt)
    print(f"[plot_gbest_y_hist] Figure saved → {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


"""
Enhanced PSO Convergence Plotting
----------------------------------
Professional styling suggestions for scientific publications.
 
Key improvements:
1. Better visual hierarchy (initial + final values annotated)
2. Percentage improvement shown
3. Adaptive y-axis formatting
4. Larger, more readable fonts for posters
5. Optional subtle shading
6. Reference line at convergence value
"""
 
def plot_gbest_y_hist_enhanced(
    log_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    negate: bool = False,
    show_improvement: bool = True,  # NEW: show percentage improvement
    show_shading: bool = True,       # NEW: toggle shaded region
) -> Path:
    """
    Enhanced publication-quality PSO convergence plot.
    
    New Parameters
    --------------
    show_improvement : bool
        If True, annotate with improvement percentage (default: True)
    show_shading : bool
        If True, add subtle shaded region below curve (default: True)
    """
    
    log_path = Path(log_path).resolve()
    gbest_y_hist = parse_gbest_y_hist(log_path)
    n_iters = len(gbest_y_hist)
 
    if negate:
        gbest_y_hist = [-v for v in gbest_y_hist]
 
    if output_path is None:
        output_path = Path.cwd() / f"{log_path.stem}_convergence_enhanced.pdf"
    output_path = Path(output_path).resolve()
 
    _apply_paper_style()
 
    # ── FIGURE ────────────────────────────────────────────────────────────────
    # Larger size for better readability on posters
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    iters = list(range(1, n_iters + 1))
 
    # IMPROVEMENT 1: Subtle shaded region (optional)
    if show_shading:
        ax.fill_between(
            iters, gbest_y_hist,
            y2=min(gbest_y_hist),  # shade to best (minimum) value
            alpha=0.06,  # very subtle
            color="#1a4e8c",
            linewidth=0,
            zorder=1,
        )
 
    # IMPROVEMENT 2: Thicker line and larger markers for posters
    ax.plot(
        iters, gbest_y_hist,
        color="#0d47a1",  # slightly deeper blue
        linewidth=2.5,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor="#0d47a1",
        zorder=3,
        label="Global Best Fitness",
        clip_on=False,  # don't clip markers at edges
    )
 
    # IMPROVEMENT 3: Annotate both initial and final values
    initial_val = gbest_y_hist[0]
    final_val = gbest_y_hist[-1]
    
    # Initial value (small, unobtrusive)
    ax.annotate(
        f"{initial_val:.3f}",
        xy=(iters[0], initial_val),
        xytext=(-15, -8),
        textcoords="offset points",
        fontsize=9.5,
        color="#0d47a1",
        ha='right',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='white',
            edgecolor='#0d47a1',
            linewidth=0.9,
            alpha=0.95
        ),
    )
    
    # IMPROVEMENT 4: Final value with improvement percentage
    improvement = ((final_val - initial_val) / abs(initial_val) * 100) if initial_val != 0 else 0
    
    if show_improvement and abs(improvement) > 0.01:
        final_text = f"{final_val:.3f}\n({improvement:+.1f}%)"
    else:
        final_text = f"{final_val:.3f}"
    
    ax.annotate(
        final_text,
        xy=(iters[-1], final_val),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=10,
        color="#0d47a1",
        ha='left',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='#e3f2fd',  # light blue background
            edgecolor='#0d47a1',
            linewidth=1.1,
            alpha=0.95
        ),
        arrowprops=dict(
            arrowstyle='->',
            color='#0d47a1',
            lw=1.4,
            connectionstyle='arc3,rad=0.2'
        ),
    )
 
    # IMPROVEMENT 5: Horizontal reference line at final value
    ax.axhline(
        y=final_val,
        color='#0d47a1',
        linestyle=':',
        linewidth=1.0,
        alpha=0.25,
        zorder=1
    )
 
    # ── AXES ──────────────────────────────────────────────────────────────────
    # IMPROVEMENT 6: Larger, bolder labels for posters/presentations
    ax.set_xlabel("Iteration", labelpad=6, fontsize=14) #, fontweight='semibold')
    ax.set_ylabel("Global Best Fitness", labelpad=6, fontsize=14) #, fontweight='semibold')
    
    if title:
        ax.set_title(title, pad=14, fontweight="bold", fontsize=15)
    
    # X-axis ticks
    ax.set_xticks(list(range(1, n_iters + 1, 2)))
    ax.set_xlim(left=0.7, right=n_iters + 0.3)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    # IMPROVEMENT 7: Adaptive y-axis formatting based on value range
    y_range = max(gbest_y_hist) - min(gbest_y_hist)
    if y_range < 0.001:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
    elif y_range < 0.01:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    elif y_range < 1:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # IMPROVEMENT 8: Better y-axis limits with breathing room
    y_min, y_max = min(gbest_y_hist), max(gbest_y_hist)
    y_margin = (y_max - y_min) * 0.12 if y_max != y_min else abs(y_max) * 0.1
    ax.set_ylim(bottom=y_min - y_margin, top=y_max + y_margin)
    
    # IMPROVEMENT 9: Larger tick labels for readability
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    fig.tight_layout()
 
    # ── SAVE ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "eps" if str(output_path).endswith(".eps") else None
    fig.savefig(output_path, format=fmt, dpi=300)  # high DPI for print quality
    print(f"[Enhanced] Figure saved → {output_path}")
 
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

    # plot_gbest_y_hist(
    #     log_path=args.log_path,
    #     output_path=args.output,
    #     title=args.title,
    #     show=args.show,
    #     negate=args.negate,
    # )
    
    plot_gbest_y_hist_enhanced(
        log_path=args.log_path,
        output_path=args.output,
        title=args.title,
        show=args.show,
        negate=args.negate,
        show_improvement=False,
        show_shading=False,
    )


if __name__ == "__main__":
    _cli()