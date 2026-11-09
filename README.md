# Heat-pump Mining Optimization Pipeline

## Overview

This project searches for a useful probabilistic automaton model of a cyber-physical system from its execution trace. It coordinates event detection and classification (discretization), event selection, and automata learning inside a particle swarm optimization (PSO) loop. Candidate configurations are evaluated by the learned model's revision score; the search minimizes the negative score, which corresponds to maximizing that score.

The implementation has two nested optimization levels:

1. **Outer PSO** searches continuous-variable event-detection settings. For each outer particle, the pipeline runs event extraction jobs for the configured signals.
2. **Inner PSO** searches which events to retain and the automata learner's parameters. It evaluates candidates by learning an automaton and scoring it against the resulting event traces.

The event traces are split into training and test data by the external event-extraction command. The learner is trained on the training traces and scored on both splits. SLURM jobs distribute the extraction and learner evaluations across a Linux cluster.

## Repository structure

```text
hpm_pipeline/
├── main.py                       # Reads configuration, starts outer PSO, reports its result
├── configs.env                   # Run settings and input/config paths
├── Algorithms/
│   ├── PSO.py                    # Outer PSO and distributed job orchestration
│   ├── InnerPSO.py               # Event-selection and learner-parameter optimization
│   ├── DecisionVar.py            # Candidate-variable representations and bounds
│   └── utils.py                  # SLURM job-state helpers
├── *job_base.sh                  # SLURM templates for extraction, preparation, and learning
├── submit.sh                     # Builds per-job scripts from a template
├── interactive.sh                # Cluster interactive-run setup
├── hpm_pipeline.sh               # Cluster batch-run setup
├── cleanup.sh                    # Removes temporary event-trace files
├── visualization.py              # Automaton rendering and PSO-log plotting utilities
├── tests/
│   ├── configs_1/ ... configs_4/ # Example property-configuration directories
│   └── ...                       # Each contains key_prop.txt and .prop files
└── outputs/                      # Default location for run-specific logs and results
```

The `tests/configs_*` directories are example property configurations; they are not an automated test suite. Job scripts under `runtime_scripts/` are assembled from the checked-in `*_job_base.sh` templates when the pipeline submits cluster jobs. The `outputs/` directory is ignored by Git.

## External project dependencies and workflow

This repository is the orchestration and optimization layer. It relies on two separate projects for the domain-specific processing:

- **[`nfer_format_compatibility`](../../Heat_Pump/nfer_format_compatibility)** performs event-trace extraction/discretization from the input time series. The job templates invoke its `event-trace-extraction-by-CPD` command, which produces event traces for the configured signals.
- **[`heat-pump-mining`](../../hpm_automata_learning/heat-pump-mining-v4)** processes the event traces: it converts and merges trace data, selects events, learns an automaton, computes revision scores, and can render an automaton. The current learner template uses its v4 build; v2 and v3 templates are also retained in this repository.

The job templates expect these projects, their required environments/builds, and cluster tools to be available in the paths and environments referenced by the scripts. In particular, the templates use SLURM commands (`sbatch`, `squeue`, and `sacct`), environment-module commands, scratch storage, and project-specific relative or absolute paths. Adapt those paths and cluster settings for a different installation; the checked-in configuration is specific to its original HPC setup.

At a high level, an outer iteration extracts event traces for candidate continuous-signal settings. For each candidate, the inner loop submits learner jobs using selected events and learner parameters, then uses their revision scores as fitness. Temporary traces are staged in cluster scratch storage; job logs and run data are written beneath the configured output directory. The current `run_for_xbest()` method is a placeholder, so the final best candidate is reported by the driver but is not currently rerun there to export a final model into `outputs/`.

## Configuration and running

`main.py` reads `configs.env` from the current working directory. The cluster shell scripts also source this file. Update its input paths and settings for your run. It includes:

- `EXEC_TRACE_FILE` and `PROP_CONFIG`: execution-trace file/directory and property-configuration directory.
- `OUTER_POP` and `OUTER_MAX_ITER`: outer PSO population and iteration limit.
- `INNER_POP` and `INNER_MAX_ITER`: inner PSO population and iteration limit.
- `SKIP_INNER_LOOP`, `SKIP_EVENT_SEL_OPT`, and `SKIP_LEARNER_OPT`: switches for bypassing optimization stages.
- `TRAIN_RATIO`, `MERGER_TYPE`, `USE_KEY_EVENT`, and `SCORING_TYPE`: trace splitting, event merging, and scoring options.

The property-configuration directory passed to `main.py` must contain:

- `key_prop.txt`: the key event/property used by event selection.
- `onetime_props.prop`: discrete properties extracted once before the outer iterations.
- `optimizing_props.prop`: continuous properties whose extraction settings are optimized. Each non-comment line identifies a property and its type, for example `zone1_temp, continuous`.

For a direct invocation from the repository directory:

```bash
python3 main.py /path/to/execution-trace /path/to/property-config \
  --output-dir /path/to/outputs
```

The `--output-dir` argument is optional and defaults to `./outputs` relative to the process working directory. Cluster runs use the provided shell templates and SLURM; `hpm_pipeline.sh` and `interactive.sh` contain cluster-specific environment setup and launch `main.py`.

## Installation

## Visualization

`visualization.py` includes utilities to plot outer-PSO fitness from a job log and to clean/render an automaton DOT file as SVG. The script's current command-line entry point renders a DOT file:

```bash
python3 visualization.py /path/to/automaton.dot
# Optionally specify the SVG destination:
python3 visualization.py /path/to/automaton.dot /path/to/automaton.svg
```

The DOT rendering command requires Graphviz's `dot` executable. The convergence plotting function is implemented in the module, but its command-line dispatch is currently commented out.

## Citation

If you use or build on this project in your research, please cite the paper:

> Simon Dierl, Nastaran Kianersi, Falk Howar, and Sean Kauffman. “Learning Probabilistic Automata from Single Continuous-Valued System Logs.” *NASA Formal Methods*, 2026, pp. 170–191. [https://doi.org/10.1007/978-3-032-28079-4_8](https://doi.org/10.1007/978-3-032-28079-4_8)

BibTeX:

```bibtex
@inbook{Dierl2026ProbabilisticAutomata,
  title     = {Learning Probabilistic Automata from Single Continuous-Valued System Logs},
  author    = {Dierl, Simon and Kianersi, Nastaran and Howar, Falk and Kauffman, Sean},
  booktitle = {NASA Formal Methods},
  year      = {2026},
  pages     = {170--191},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-28079-4_8}
}
```

## TODO

- [ ] Add a documented sample `configs.env` with safe, portable example values.
- [ ] Add or link a small sample execution-trace input and document its expected format.
- [ ] Add representative outputs, including job logs, event traces, scores, and a learned automaton, with a description of where each is written.
