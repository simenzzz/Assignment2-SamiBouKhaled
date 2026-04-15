# Parallel Reinforcement Learning

Grid World reinforcement learning agent using **State-Action Counting + Reward Aggregation** with epsilon-greedy exploration (epsilon = 0.1). Three implementations:

1. **Sequential** (single-threaded C)
2. **OpenMP** (multi-threaded with thread-local tables merged after execution)
3. **MPI** (multi-process with `MPI_Reduce` aggregation)

Each runs 1000 episodes across 4x4, 8x8, and 16x16 grids and outputs CSV data for analysis.

## How to Build

**Prerequisites:**
- GCC with OpenMP support
- OpenMPI (`openmpi-bin`, `libopenmpi-dev` on Debian/Ubuntu)

```bash
make
```

This builds three executables in `compiled/`: `sequential`, `parallel_omp`, `parallel_mpi`.

To clean up:
```bash
make clean
```

## How to Run

All programs output CSV to stdout. Redirect to files in `results/`.

```bash
mkdir -p results
```

**Sequential** — takes grid size (4, 8, or 16):
```bash
compiled/sequential 4 > results/seq_4.csv
compiled/sequential 8 > results/seq_8.csv
compiled/sequential 16 > results/seq_16.csv
```

**OpenMP** — takes grid size and thread count:
```bash
compiled/parallel_omp 4 1 > results/omp_4_t1.csv
compiled/parallel_omp 4 2 > results/omp_4_t2.csv
compiled/parallel_omp 4 4 > results/omp_4_t4.csv

compiled/parallel_omp 8 1 > results/omp_8_t1.csv
compiled/parallel_omp 8 2 > results/omp_8_t2.csv
compiled/parallel_omp 8 4 > results/omp_8_t4.csv

compiled/parallel_omp 16 1 > results/omp_16_t1.csv
compiled/parallel_omp 16 2 > results/omp_16_t2.csv
compiled/parallel_omp 16 4 > results/omp_16_t4.csv
```

**MPI** — takes grid size; process count set via `mpirun -np`:
```bash
mpirun --allow-run-as-root -np 1 compiled/parallel_mpi 4 > results/mpi_4_p1.csv
mpirun --allow-run-as-root -np 2 compiled/parallel_mpi 4 > results/mpi_4_p2.csv
mpirun --allow-run-as-root -np 4 compiled/parallel_mpi 4 > results/mpi_4_p4.csv

mpirun --allow-run-as-root -np 1 compiled/parallel_mpi 8 > results/mpi_8_p1.csv
mpirun --allow-run-as-root -np 2 compiled/parallel_mpi 8 > results/mpi_8_p2.csv
mpirun --allow-run-as-root -np 4 compiled/parallel_mpi 8 > results/mpi_8_p4.csv

mpirun --allow-run-as-root -np 1 compiled/parallel_mpi 16 > results/mpi_16_p1.csv
mpirun --allow-run-as-root -np 2 compiled/parallel_mpi 16 > results/mpi_16_p2.csv
mpirun --allow-run-as-root -np 4 compiled/parallel_mpi 16 > results/mpi_16_p4.csv
```

> Note: `--allow-run-as-root` is only needed when running as root. Remove it otherwise.

## How to Visualize

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Run the analysis script from the project root:
```bash
python3 scripts/analysis.py
```

This reads all CSV files from `results/` and generates:
- `results/rewards_{N}x{N}.png` — episode reward plots per grid size
- `results/opt_action_pct_{N}x{N}.png` — optimal action percentage per state
- `results/speedup.png` — speedup and efficiency charts
- `results/tables.md` — summary tables in Markdown

## Project Structure

```
lab2/
├── Makefile           # Build configuration
├── README.md          # This file
├── requirements.txt   # Python dependencies
├── .gitignore
├── src/
│   ├── sequential.c       # Sequential implementation
│   ├── parallel_omp.c     # OpenMP parallel implementation
│   └── parallel_mpi.c     # MPI parallel implementation
├── scripts/
│   └── analysis.py        # Plotting and analysis script
├── compiled/              # Compiled binaries 
└── results/               # Generated results 
```
