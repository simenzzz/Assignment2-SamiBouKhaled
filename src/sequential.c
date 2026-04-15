#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

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
int cnt[s][a];
double rew_sum[s][a];
double ep_rew[n_episodes];

// grid setup
void init_grid(int size) {
    grid_size = size;
    int total_cells = size * size;

    // intialize to zeros (empty state)
    for (int i = 0; i < total_cells; i++) {
        grid[i] = EMPTY;
    }

    if (size == 4) {
        // 4x4 grid
        grid[2] = OBSTACLE;
        grid[5] = OBSTACLE;
        grid[15] = GOAL;
        grid[11] = TRAP;
    }
    else if (size == 8) {
        // 8x8 grid
        grid[10] = OBSTACLE;
        grid[21] = OBSTACLE;
        grid[42] = OBSTACLE;
        grid[53] = OBSTACLE;
        grid[63] = GOAL;
        grid[58] = TRAP;
        grid[62] = TRAP;
    }
    else if (size == 16) {
        // 16x16 grid
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

// try to move in a direction, return new state
int move(int state, int action) {
    int n = state / grid_size;
    int m = state % grid_size;
    int new_n = n;
    int new_m = m;

    if (action == UP) {
        new_n = n - 1;
    }
    else if (action == DOWN) {
        new_n = n + 1;
    }
    else if (action == LEFT) {
        new_m = m - 1;
    }
    else if (action == RIGHT) {
        new_m = m + 1;
    }

    // check boundaries
    if (new_n < 0 || new_n >= grid_size || new_m < 0 || new_m >= grid_size) {
        return state; // stay in place
    }

    int new_state = new_n * grid_size + new_m;

    // check obstacle
    if (grid[new_state] == OBSTACLE) {
        return state; // stay in place
    }

    return new_state;
}

// reward computations
double get_reward(int state) {
    if (grid[state] == GOAL) {
        return 1.0;
    }
    else if (grid[state] == TRAP) {
        return -1.0;
    }
    else {
        // random penalty between -0.1 and -0.05,
        double r = (double)rand() / RAND_MAX; // always between 0.0 and 1.0
        return -0.1 + r * 0.05; // coerce to -0.1 <-> -0.05
    }
}

// pick an action using epsilon-greedy
int choose_action(int state) {
    double r = (double)rand() / RAND_MAX;

    // (P(r < 0.1) = 0.1) so pick random action if thats the case
    if (r < EPSILON) {
        return rand() % a; // (0,1,2,3)
    }

    // else: pick action with best average reward
    int best_action = 0;
    double best_avg = -9999.0;

    for (int k = 0; k < a; k++) {
        double avg;
        if (cnt[state][k] > 0) {
            avg = rew_sum[state][k] / cnt[state][k];
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

// pick a random starting position (not obstacle, goal, or trap)
int random_start() {
    int total = grid_size * grid_size;
    int state = rand() % total;
    while (grid[state] != EMPTY){
        state = rand() % total;
    }
    return state;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("invalid args");
        return 1;
    }

    int size = atoi(argv[1]);
    if (size != 4 && size != 8 && size != 16) {
        printf("invalid grid size");
        return 1;
    }

    srand(time(NULL));
    init_grid(size);

    int total_states = size * size;

    // run the full training loop 20 (timing_runs) times and average wall-clock.
    // each repetition is an independent 1000-episode training run so the
    // final tables reflect only the last repetition.
    double t_total = 0.0;

    for (int rep = 0; rep < timing_runs; rep++) {
        memset(cnt, 0, sizeof(cnt));
        memset(rew_sum, 0, sizeof(rew_sum));

        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);

        for (int ep = 0; ep < n_episodes; ep++) {
            int state = random_start();
            double total_reward = 0.0;

            for (int step = 0; step < max_steps; step++) {
                int action = choose_action(state);
                int next_state = move(state, action);
                double reward = get_reward(next_state);

                cnt[state][action] += 1;
                rew_sum[state][action] += reward;
                total_reward += reward;

                state = next_state;

                // stop if we reached goal or trap
                if (grid[state] == GOAL || grid[state] == TRAP) {
                    break;
                }
            }

            ep_rew[ep] = total_reward;
        }

        gettimeofday(&end_time, NULL);
        t_total += (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    }

    double avg_t = t_total / timing_runs;

    // compute average reward and optimal policy
    double avg_reward[s][a];
    int optimal_action[s];

    for (int i = 0; i < total_states; i++) {
        double best_avg = -9999.0;
        int best_act = 0;

        for (int j = 0; j < a; j++) {
            if (cnt[i][j] > 0) {
                avg_reward[i][j] = rew_sum[i][j] / cnt[i][j];
            }
            else {
                avg_reward[i][j] = 0.0;
            }

            if (avg_reward[i][j] > best_avg) {
                best_avg = avg_reward[i][j];
                best_act = j;
            }
        }

        optimal_action[i] = best_act; // good enough
    }

    // print results
    printf("TIME,%.6f\n", avg_t);

    // episode rewards
    printf("EPISODE_REWARDS\n");
    for (int ep = 0; ep < n_episodes; ep++) {
        printf("%d,%.6f\n", ep, ep_rew[ep]);
    }
    printf("END_EPISODE_REWARDS\n");

    // optimal policy
    printf("POLICY\n");
    const char *action_names[] = {"UP", "DOWN", "LEFT", "RIGHT"};
    for (int i = 0; i < total_states; i++) {
        printf("%d,%s\n", i, action_names[optimal_action[i]]);
    }
    printf("END_POLICY\n");

    // average reward per state-action pair
    printf("AVG_REWARD\n");
    for (int i = 0; i < total_states; i++) {
        printf("%d,%.6f,%.6f,%.6f,%.6f\n", i,
               avg_reward[i][0], avg_reward[i][1],
               avg_reward[i][2], avg_reward[i][3]);
    }
    printf("END_AVG_REWARD\n");

    // per-state stats: visits, optimal-action count, percentage
    // visits[s] = sum over a of cnt[s][a]
    // a* = optimal_action[s]
    // OptimalActionPercentage[s] = cnt[s][a*] / visits[s]
    printf("STATE_STATS\n");
    for (int i = 0; i < total_states; i++) {
        int visits = 0;
        for (int j = 0; j < a; j++) {
            visits += cnt[i][j];
        }
        int opt_count = cnt[i][optimal_action[i]];
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
