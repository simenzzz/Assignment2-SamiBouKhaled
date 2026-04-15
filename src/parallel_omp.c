#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

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
int gcnt[s][a];
double grew[s][a];
double episode_rewards[n_episodes];

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

double get_reward(int state, unsigned int *seed) {
    if (grid[state] == GOAL) {
        return 1.0;
    }
    else if (grid[state] == TRAP) {
        return -1.0;
    }
    else {
        double r = (double)rand_r(seed) / RAND_MAX;
        return -0.1 + r * 0.05;
    }
}

// epsilon-greedy using local tables
int choose_action(int state, int lc[][a],
                  double lr[][a], unsigned int *seed) {
    double r = (double)rand_r(seed) / RAND_MAX;

    if (r < EPSILON) {
        return rand_r(seed) % a;
    }

    int best_action = 0;
    double best_avg = -9999.0;

    for (int k = 0; k < a; k++) {
        double avg;
        if (lc[state][k] > 0) {
            avg = lr[state][k] / lc[state][k];
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

int random_start(unsigned int *seed) {
    int total = grid_size * grid_size;
    int state;
    do {
        state = rand_r(seed) % total;
    } while (grid[state] != EMPTY);
    return state;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <grid_size> <num_threads>\n", argv[0]);
        printf("  grid_size: 4, 8, or 16\n");
        printf("  num_threads: 1, 2, or 4\n");
        return 1;
    }

    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (size != 4 && size != 8 && size != 16) {
        printf("Error: grid_size must be 4, 8, or 16\n");
        return 1;
    }

    init_grid(size);
    omp_set_num_threads(num_threads);

    int total_states = size * size;

    double t_total = 0.0;

    for (int rep = 0; rep < timing_runs; rep++) {
        memset(gcnt, 0, sizeof(gcnt));
        memset(grew, 0, sizeof(grew));

        double start_time = omp_get_wtime();

        // each thread does its share of episodes
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            int nt = omp_get_num_threads();
            unsigned int seed = (unsigned int)(time(NULL) ^ (t * 2654435761u));

            // thread-local tables
            int lc[s][a];
            double lr[s][a];
            memset(lc, 0, sizeof(lc));
            memset(lr, 0, sizeof(lr));

            int n_ep = n_episodes / nt;
            int ep_start = t * n_ep;
            int ep_end = (t == nt - 1) ? n_episodes : ep_start + n_ep;

            for (int ep = ep_start; ep < ep_end; ep++) {
                int state = random_start(&seed);
                double total_reward = 0.0;

                for (int step = 0; step < max_steps; step++) {
                    int action = choose_action(state, lc, lr, &seed);
                    int next_state = move(state, action);
                    double reward = get_reward(next_state, &seed);

                    lc[state][action] += 1;
                    lr[state][action] += reward;
                    total_reward += reward;

                    state = next_state;

                    if (grid[state] == GOAL || grid[state] == TRAP) {
                        break;
                    }
                }

                episode_rewards[ep] = total_reward;
            }

            // add up all the local tables
            #pragma omp critical
            {
                for (int i = 0; i < total_states; i++) {
                    for (int j = 0; j < a; j++) {
                        gcnt[i][j] += lc[i][j];
                        grew[i][j] += lr[i][j];
                    }
                }
            }
        }

        double end_time = omp_get_wtime();
        t_total += end_time - start_time;
    }

    double avg_t = t_total / timing_runs;

    // compute optimal policy from merged tables
    double avg_reward[s][a];
    int optimal_action[s];

    for (int i = 0; i < total_states; i++) {
        double best_avg = -9999.0;
        int best_a = 0;

        for (int j = 0; j < a; j++) {
            if (gcnt[i][j] > 0) {
                avg_reward[i][j] = grew[i][j] / gcnt[i][j];
            }
            else {
                avg_reward[i][j] = 0.0;
            }

            if (avg_reward[i][j] > best_avg) {
                best_avg = avg_reward[i][j];
                best_a = j;
            }
        }

        optimal_action[i] = best_a;
    }

    // print results
    printf("TIME,%.6f\n", avg_t);

    printf("EPISODE_REWARDS\n");
    for (int ep = 0; ep < n_episodes; ep++) {
        printf("%d,%.6f\n", ep, episode_rewards[ep]);
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
            visits += gcnt[i][j];
        }
        int opt_count = gcnt[i][optimal_action[i]];
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

    return 0;
}
