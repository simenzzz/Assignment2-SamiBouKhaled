#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define s 256
#define a 4
#define n_episodes 1000
#define max_steps 50
#define EPSILON 0.1
#define timing_runs 20

// cell types
#define EMPTY 0
#define OBSTACLE 1
#define GOAL 2
#define TRAP 3

// actions
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

int grid_size;
int grid[s];

void init_grid(int size) {
    grid_size = size;
    int total = size * size;

    for (int i = 0; i < total; i++) {
        grid[i] = EMPTY;
    }

    if (size == 4) {
        grid[2] = OBSTACLE;
        grid[5] = OBSTACLE;
        grid[15] = GOAL;
        grid[11] = TRAP;
    }
    else if (size == 8) {
        grid[10] = OBSTACLE;
        grid[21] = OBSTACLE;
        grid[42] = OBSTACLE;
        grid[53] = OBSTACLE;
        grid[63] = GOAL;
        grid[58] = TRAP;
        grid[62] = TRAP;
    }
    else if (size == 16) {
        grid[18] = OBSTACLE;
        grid[37] = OBSTACLE;
        grid[74] = OBSTACLE;
        grid[93] = OBSTACLE;
        grid[122] = OBSTACLE;
        grid[157] = OBSTACLE;
        grid[198] = OBSTACLE;
        grid[221] = OBSTACLE;
        grid[255] = GOAL;
        grid[238] = TRAP;
        grid[251] = TRAP;
        grid[254] = TRAP;
    }
}

int move(int state, int action) {
    int row = state / grid_size;
    int col = state % grid_size;
    int new_row = row;
    int new_col = col;

    if (action == UP) {
        new_row = row - 1;
    }
    else if (action == DOWN) {
        new_row = row + 1;
    }
    else if (action == LEFT) {
        new_col = col - 1;
    }
    else if (action == RIGHT) {
        new_col = col + 1;
    }

    if (new_row < 0 || new_row >= grid_size || new_col < 0 || new_col >= grid_size) {
        return state;
    }

    int new_state = new_row * grid_size + new_col;

    if (grid[new_state] == OBSTACLE) {
        return state;
    }

    return new_state;
}

double get_reward(int state) {
    if (grid[state] == GOAL) {
        return 1.0;
    }
    else if (grid[state] == TRAP) {
        return -1.0;
    }
    else {
        double r = (double)rand() / RAND_MAX;
        return -0.1 + r * 0.05;
    }
}

int choose_action(int state, int my_cnt[][a],
                  double my_rew[][a]) {
    double r = (double)rand() / RAND_MAX;

    if (r < EPSILON) {
        return rand() % a;
    }

    int best_action = 0;
    double best_avg = -9999.0;

    for (int k = 0; k < a; k++) {
        double avg;
        if (my_cnt[state][k] > 0) {
            avg = my_rew[state][k] / my_cnt[state][k];
        }
        else {
            avg = 0.0;
        }

        if (avg > best_avg) {
            best_avg = avg;
            best_action = k;
        }
    }

    return best_action;
}

int random_start() {
    int total = grid_size * grid_size;
    int state;
    do {
        state = rand() % total;
    } while (grid[state] != EMPTY);
    return state;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <procs> %s <grid_size>\n", argv[0]);
            printf("  grid_size: 4, 8, or 16\n");
        }
        MPI_Finalize();
        return 1;
    }

    int size = atoi(argv[1]);
    if (size != 4 && size != 8 && size != 16) {
        if (rank == 0) {
            printf("Error: grid_size must be 4, 8, or 16\n");
        }
        MPI_Finalize();
        return 1;
    }

    init_grid(size);

    int total_states = size * size;

    // each process keeps its own counts
    int my_cnt[s][a];
    double my_rew[s][a];

    // split episodes evenly
    // TODO: handle case where n_episodes not divisible by num_procs
    int n_ep = n_episodes / num_procs;
    int ep_start = rank * n_ep;
    int ep_end = (rank == num_procs - 1) ? n_episodes : ep_start + n_ep;
    int my_ep = ep_end - ep_start;

    double my_ep_rew[n_episodes];

    // aggregate tables (filled by MPI_Reduce)
    int g_cnt[s][a];
    double g_rew[s][a];
    double all_ep[n_episodes];

    double t_total = 0.0;
    double avg_t = 0.0;

    for (int rep = 0; rep < timing_runs; rep++) {
        srand(time(NULL) + rank * 1000 + rep * 17);

        memset(my_cnt, 0, sizeof(my_cnt));
        memset(my_rew, 0, sizeof(my_rew));
        memset(my_ep_rew, 0, sizeof(my_ep_rew));

        // wait for everyone before timing
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        for (int ep = 0; ep < my_ep; ep++) {
            int state = random_start();
            double total_reward = 0.0;

            for (int step = 0; step < max_steps; step++) {
                int action = choose_action(state, my_cnt, my_rew);
                int next_state = move(state, action);
                double reward = get_reward(next_state);

                my_cnt[state][action] += 1;
                my_rew[state][action] += reward;
                total_reward += reward;

                state = next_state;

                if (grid[state] == GOAL || grid[state] == TRAP) {
                    break;
                }
            }

            my_ep_rew[ep_start + ep] = total_reward;
        }

        // sum everything up to rank 0
        MPI_Reduce(my_cnt, g_cnt, s * a,
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(my_rew, g_rew, s * a,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(my_ep_rew, all_ep, n_episodes,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        double end_time_val = MPI_Wtime();
        double elapsed = end_time_val - start_time;

        // slowest-rank wall time for this repetition
        double rep_max;
        MPI_Reduce(&elapsed, &rep_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            t_total += rep_max;
        }
    }

    if (rank == 0) {
        avg_t = t_total / timing_runs;
    }

    // rank 0 computes policy and prints results
    if (rank == 0) {
        double avg_reward[s][a];
        int optimal_action[s];

        for (int i = 0; i < total_states; i++) {
            double best_avg = -9999.0;
            int best_act = 0;

            for (int j = 0; j < a; j++) {
                if (g_cnt[i][j] > 0) {
                    avg_reward[i][j] = g_rew[i][j] / g_cnt[i][j];
                }
                else {
                    avg_reward[i][j] = 0.0;
                }

                if (avg_reward[i][j] > best_avg) {
                    best_avg = avg_reward[i][j];
                    best_act = j;
                }
            }

            optimal_action[i] = best_act;
        }

        printf("TIME,%.6f\n", avg_t);

        printf("EPISODE_REWARDS\n");
        for (int ep = 0; ep < n_episodes; ep++) {
            printf("%d,%.6f\n", ep, all_ep[ep]);
        }
        printf("END_EPISODE_REWARDS\n");

        printf("POLICY\n");
        const char *action_names[] = {"UP", "DOWN", "LEFT", "RIGHT"};
        for (int i = 0; i < total_states; i++) {
            printf("%d,%s\n", i, action_names[optimal_action[i]]);
        }
        printf("END_POLICY\n");

        printf("AVG_REWARD\n");
        for (int i = 0; i < total_states; i++) {
            printf("%d,%.6f,%.6f,%.6f,%.6f\n", i,
                   avg_reward[i][0], avg_reward[i][1],
                   avg_reward[i][2], avg_reward[i][3]);
        }
        printf("END_AVG_REWARD\n");

        printf("STATE_STATS\n");
        for (int i = 0; i < total_states; i++) {
            int visits = 0;
            for (int j = 0; j < a; j++) {
                visits += g_cnt[i][j];
            }
            int opt_count = g_cnt[i][optimal_action[i]];
            double pct;
            if (visits > 0) {
                pct = 100.0 * (double)opt_count / (double)visits;
            }
            else {
                pct = 0.0;
            }
            printf("%d,%s,%d,%d,%.4f\n", i, action_names[optimal_action[i]],
                   visits, opt_count, pct);
        }
        printf("END_STATE_STATS\n");

    }

    MPI_Finalize();
    return 0;
}
