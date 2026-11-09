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
from datetime import datetime
import subprocess
import sys

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


# ── File Parsers ───────────────────────────────────────────────────────────────

# --- Log Parser for gbest_y history ---
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

    return gbest_y_hist #[:-1]

# --- Log Parser for execution time ---
def calculate_exec_time(log_path: str | Path) -> float:
    """
    Calculate total execution time from a PSO log file by parsing lines 
    containing "Pipeline job started at: " (example: at: Mon May 25 06:28:30 PM EDT 2026)
    and "Pipeline job ended at: " (example: at: Mon May 25 09:08:59 PM EDT 2026) 
    Calculate the difference in hh:mm:ss and print it in this format
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    start_time_str = None
    end_time_str = None

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if "Pipeline job started at:" in line:
                start_time_str = line.split("at:")[1].strip()
            elif "Pipeline job ended at:" in line:
                end_time_str = line.split("at:")[1].strip()

    if not start_time_str or not end_time_str:
        print("Could not find start or end time in the log.")
        return 0.0

    time_format = "%a %b %d %I:%M:%S %p %Z %Y"
    start_time = datetime.strptime(start_time_str, time_format)
    end_time = datetime.strptime(end_time_str, time_format)

    exec_time_seconds = (end_time - start_time).total_seconds()
    exec_time_hms = str(end_time - start_time)

    print(f"Total execution time: {exec_time_hms} (hh:mm:ss)")

    return exec_time_seconds
    
# --- Dot file parser and cleaner for automaton visualization ---
# def clean_dot(src: str) -> str:
#     # 1. circle -> ellipse so long labels fit snugly
#     src = src.replace('shape="circle"', 'shape="ellipse"')
 
#     def transform_label(m: re.Match) -> str:
#         label = m.group(1)
#         # 2. drop "_std<number>" from event names (e.g. _std0.02)
#         label = re.sub(r"_std[\d.]+", "", label)
#         # 3. drop the " / tSD=..." segment entirely
#         label = re.sub(r" / tSD=[^\"]*", "", label)
#         # 4. drop the leading "- / " from edge labels like "- / p=0.50"
#         label = re.sub(r"^- / ", "", label)
#         # 5. drop the leading "c/" or "d/" prefix from event names
#         label = re.sub(r"^[cd]/", "", label)
#         # 6. turn remaining " / " separators into newlines for tighter ellipses
#         label = label.replace(" / ", r"\n")
#         return f'label="{label}"'
 
#     return re.sub(r'label="([^"]*)"', transform_label, src)

# --- Dot file parser and cleaner for automaton visualization ---
def clean_dot(src: str) -> str:
    # Each entry: (regex pattern, replacement string).
    # Patterns are matched anywhere in the label unless anchored with ^ or $.
    # Replacements may contain HTML tags like <sub>1</sub> or <sup>2</sup>;
    # the script will switch that node to an HTML-like label automatically.
    RENAMES = [
        (r"PLANT_INFO_CurrentIL", r"I<sub>L</sub>"),
        (r"PLANT_INFO_CurrentVC", r"V<sub>C</sub>"),
        (r"CTRL_INFO_ModeSetting", r"Ctrl-Mode"),
    ]

    HTML_TAG_RE = re.compile(r"<(sub|sup|b|i|u|font|br)\b", re.IGNORECASE)
    
    # 1. circle -> ellipse so long labels fit snugly
    src = src.replace('shape="circle"', 'shape="ellipse"')
 
    def transform_label(m: re.Match) -> str:
        label = m.group(1)
        # 2. drop "_std<number>" from event names (e.g. _std0.02)
        label = re.sub(r"_std[\d.]+", "", label)
        # 3. drop the " / tSD=..." segment entirely
        label = re.sub(r" / tSD=[^\"]*", "", label)
        # 4. drop the leading "- / " from edge labels like "- / p=0.50"
        label = re.sub(r"^- / ", " ", label)
        # 5. drop the leading "c/" or "d/" prefix
        label = re.sub(r"^[cd]/", "", label)
        # 6. apply user-defined renames
        for pattern, repl in RENAMES:
            label = re.sub(pattern, repl, label)
 
        # 7. emit the label in the right syntax
        if HTML_TAG_RE.search(label):
            # HTML-like label: <...> wrapper, <br/> as newline
            label = label.replace(" / ", "<br/>")
            return f"label=<{label}>"
        else:
            # ordinary quoted label: \n as newline
            label = label.replace(" / ", r"\n")
            return f'label="{label}"'
 
    # return re.sub(r'label="([^"]*)"', transform_label, src)
    
    src = re.sub(r'label="([^"]*)"', transform_label, src)
 
    # 8. on edge lines (containing "->"), rename label= to xlabel= so
    #    Graphviz's collision-avoiding label placer is used instead of
    #    inline placement. Node labels stay as label=.
    out_lines = []
    for line in src.splitlines():
        if "->" in line and re.search(r'\blabel\s*=', line):
            line = re.sub(r'\blabel(\s*=)', r"xlabel\1", line, count=1)
        out_lines.append(line)
    return "\n".join(out_lines)


# ── Plotter ──────────────────────────────────────────────────────────────────

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
        # output_path = Path.cwd() / f"{log_path.stem}_convergence_enhanced.pdf"
        output_path = log_path.parent / f"{log_path.stem}_convergence_enhanced.svg"
    output_path = Path(output_path).resolve()
    print(output_path)
    
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
        xytext=(0, 14),
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
    # fmt = "eps" if str(output_path).endswith(".eps") else "svg"
    fmt = output_path.suffix.lstrip(".")
    # if not fmt:
        # fmt = "svg"
    fig.savefig(output_path, format=fmt, dpi=300)  # high DPI for print quality
    print(f"[Enhanced] Figure saved → {output_path}")
 
    if show:
        plt.show()
 
    plt.close(fig)
    return output_path

# ── CLI ───────────────────────────────────────────────────────────────────────

""" Basic Usage:
    python visualization.py path/to/job.out
    Produces path/to/job_convergence_enhanced.svg with the convergence plot.
"""
def convergence_plot() -> None:
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
        "--negate", action="store_false",
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
        show_improvement=True,
        show_shading=False,
    )
    
    calculate_exec_time(log_path=args.log_path)

""" Basic Usage:
    python visualization.py path/to/automaton.dot
    Produces path/to/automaton_cleaned.dot and path/to/automaton_cleaned.svg
"""
def render_automaton() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("Usage: python3 clean_automaton.py input.dot [output.svg]")
 
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) == 3 else in_path.with_suffix(".svg")
    cleaned_path = in_path.with_name(in_path.stem + "_cleaned.dot")
 
    cleaned_path.write_text(clean_dot(in_path.read_text()))
 
    subprocess.run(
        [
            "dot", str(cleaned_path), "-Tsvg",
            "-Nfontsize=10", "-Efontsize=9",
            "-Gsize=7.5,10", "-Gratio=compress",
            "-Granksep=0.7", "-Gnodesep=0.8",
            "-Gforcelabels=true",
            "-o", str(out_path),
        ],

        check=True,
    )
 
    print(f"Cleaned dot file: {cleaned_path}")
    print(f"Rendered SVG:    {out_path}")


if __name__ == "__main__":
    # convergence_plot()
    render_automaton()