CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O2
OMP_FLAGS = -fopenmp
BINDIR = compiled

all: $(BINDIR)/sequential $(BINDIR)/parallel_omp $(BINDIR)/parallel_mpi

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/sequential: src/sequential.c | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BINDIR)/parallel_omp: src/parallel_omp.c | $(BINDIR)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $< -lm

$(BINDIR)/parallel_mpi: src/parallel_mpi.c | $(BINDIR)
	$(MPICC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -rf $(BINDIR)

.PHONY: all clean
