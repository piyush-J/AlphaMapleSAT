#include <bits/stdc++.h>
using namespace std;

/**
 * Simplified C++ version of AlphaMapleSAT cubing.
 * This implementation focuses on argument parsing, CNF parsing,
 * a basic unit propagation based on the provided BCP routines,
 * and a depth-first search that explores variable assignments.
 * It does not implement the full MCTS algorithm but mimics the
 * cube generation logic of the Python tool.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>

// ----------------- BCP ROUTINES -----------------
// The following code is adapted from the provided BCP snippet.

#define MAX_VARS 20000
#define MAX_CLAUSES 1000000
#define MAX_LITS 1000000
#define BIMP_GROW_CHUNK 4
#define ASSIGN_NONE 0
#define ASSIGN_TRUE 1
#define ASSIGN_FALSE 2

int n_vars, n_clauses;
int *clauses[MAX_CLAUSES];
int clause_sizes[MAX_CLAUSES];
int clause_stamp[MAX_CLAUSES];
int *watch1[MAX_CLAUSES];
int *watch2[MAX_CLAUSES];

uint8_t assignments[MAX_VARS + 1];
int current_stamp = 1;

// Binary implication arrays
int *bimp[MAX_VARS * 2 + 2];
int bimp_size[MAX_VARS * 2 + 2];
int bimp_capacity[MAX_VARS * 2 + 2];

int var_activity[MAX_VARS + 1];
int global_queue[MAX_VARS + 1];

static inline int lit_index(int lit) {
    return (lit > 0) ? lit : n_vars - lit;
}

static inline void reset_assignments() {
    memset(assignments, ASSIGN_NONE, sizeof(uint8_t) * (n_vars + 1));
    current_stamp++;
}

void add_bimp(int lit, int implied) {
    int idx = lit_index(lit);
    if (bimp_capacity[idx] == 0) {
        bimp_capacity[idx] = BIMP_GROW_CHUNK;
        bimp[idx] = (int*)malloc(sizeof(int) * bimp_capacity[idx]);
    } else if (bimp_size[idx] >= bimp_capacity[idx]) {
        bimp_capacity[idx] *= 2;
        bimp[idx] = (int*)realloc(bimp[idx], sizeof(int) * bimp_capacity[idx]);
    }
    bimp[idx][bimp_size[idx]++] = implied;
}

void parse_cnf(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        exit(1);
    }
    char line[10000];

    int clause_index = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'p') {
            sscanf(line, "p cnf %d %d", &n_vars, &n_clauses);
        } else if (line[0] != 'c') {
            int lits[1000], size = 0, lit;
            char *ptr = line;
            while (sscanf(ptr, "%d", &lit) == 1 && lit != 0) {
                lits[size++] = lit;
                ptr = strchr(ptr, ' ');
                if (!ptr) break;
                ptr++;
            }
            clauses[clause_index] = (int *)malloc(sizeof(int) * size);
            memcpy(clauses[clause_index], lits, sizeof(int) * size);
            clause_sizes[clause_index] = size;
            clause_stamp[clause_index] = 0;
            if (size >= 2) {
                watch1[clause_index] = &clauses[clause_index][0];
                watch2[clause_index] = &clauses[clause_index][1];
            }
            if (size == 2) {
                add_bimp(-lits[0], lits[1]);
                add_bimp(-lits[1], lits[0]);
            }
            if (size <= 3) {
                for (int j = 0; j < size; j++)
                    var_activity[abs(lits[j])]++;
            }
            clause_index++;
        }
    }
    n_clauses = clause_index;
    fclose(fp);
}

int propagate_bimp(int lit, std::vector<int>& propagated) {
    int front = 0, rear = 0;
    global_queue[rear++] = lit;

    while (front < rear) {
        int l = global_queue[front++];
        int var = abs(l);
        if (assignments[var] != ASSIGN_NONE) {
            if ((assignments[var] == ASSIGN_TRUE && l < 0) ||
                (assignments[var] == ASSIGN_FALSE && l > 0))
                return 0; // conflict
            continue;
        }
        assignments[var] = (l > 0) ? ASSIGN_TRUE : ASSIGN_FALSE;
        propagated.push_back(var);

        int idx = lit_index(l);
        for (int i = 0; i < bimp_size[idx]; i++) {
            global_queue[rear++] = bimp[idx][i];
        }
    }
    return 1;
}

int propagate_big_clauses(std::vector<int>& propagated) {
    for (int i = 0; i < n_clauses; i++) {
        if (clause_sizes[i] <= 2 || clause_stamp[i] == current_stamp) continue;
        clause_stamp[i] = current_stamp;

        int *lits = clauses[i];
        int sat = 0, unassigned = 0, last_unassigned = 0;

        for (int j = 0; j < clause_sizes[i]; j++) {
            int lit = lits[j];
            int var = abs(lit);
            if (assignments[var] == ASSIGN_NONE) {
                unassigned++;
                last_unassigned = lit;
            } else if ((assignments[var] == ASSIGN_TRUE && lit > 0) ||
                       (assignments[var] == ASSIGN_FALSE && lit < 0)) {
                sat = 1;
                break;
            }
        }

        if (!sat && unassigned == 0) return 0;
        if (!sat && unassigned == 1) {
            int v = abs(last_unassigned);
            assignments[v] = (last_unassigned > 0) ? ASSIGN_TRUE : ASSIGN_FALSE;
            propagated.push_back(v);
        }
    }
    return 1;
}

int unit_propagation(int lit, std::vector<int>& propagated) {
    propagated.clear();
    if (!propagate_bimp(lit, propagated)) return 0;
    return propagate_big_clauses(propagated);
}

// --------------- END BCP ROUTINES ---------------

struct Options {
    std::string filename;
    int n_cutoff = -1;
    int d_cutoff = -1;
    int m_vars = 0;
    std::string out_file;
    int num_sims = 10;
};

/** Parse command line arguments.
 * Only minimal checks are performed. */
Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i+1 < argc) {
            opt.n_cutoff = atoi(argv[++i]);
        } else if (arg == "-d" && i+1 < argc) {
            opt.d_cutoff = atoi(argv[++i]);
        } else if (arg == "-m" && i+1 < argc) {
            opt.m_vars = atoi(argv[++i]);
        } else if (arg == "-o" && i+1 < argc) {
            opt.out_file = argv[++i];
        } else if (arg == "-numMCTSSims" && i+1 < argc) {
            opt.num_sims = atoi(argv[++i]);
        } else if (arg[0] != '-') {
            opt.filename = arg;
        }
    }
    return opt;
}

// Store a single cube
struct Cube {
    std::vector<int> lits;
};

int node_count = 0;

// Basic DFS that explores assignments up to depth/n_cutoff
void dfs(std::vector<int>& path, int depth, int& count, Options& opt,
         const std::vector<int>& vars, std::vector<Cube>& cubes) {
    node_count++;
    if ((opt.d_cutoff != -1 && depth >= opt.d_cutoff) ||
        (opt.n_cutoff != -1 && (int)path.size() >= opt.n_cutoff)) {
        cubes.push_back({path});
        return;
    }
    if (depth >= (int)vars.size()) {
        cubes.push_back({path});
        return;
    }
    int var = vars[depth];
    for (int val = 0; val < 2; ++val) {
        int lit = val ? var : -var;
        path.push_back(lit);
        count++;
        dfs(path, depth + 1, count, opt, vars, cubes);
        path.pop_back();
    }
}

// Score variables using propagation as in the provided snippet
std::vector<int> preselect_vars(int M) {
    std::vector<int> selected;
    int scores[MAX_VARS + 1][2] = {0};

    for (int v = 1; v <= n_vars && v <= M; v++) {
        std::vector<int> propagated;
        int pos = 0, neg = 0;

        reset_assignments();
        if (unit_propagation(v, propagated)) pos = propagated.size();

        reset_assignments();
        if (unit_propagation(-v, propagated)) neg = propagated.size();

        scores[v][0] = v;
        scores[v][1] = pos * neg + 10 * (pos + neg) +
                       100 * (bimp_size[lit_index(v)] + bimp_size[lit_index(-v)]) +
                       var_activity[v] * 5;
    }

    std::vector<std::pair<int,int>> vec;
    for (int v = 1; v <= n_vars && v <= M; ++v) {
        if (scores[v][1] > 0) vec.push_back({scores[v][1], v});
    }
    std::sort(vec.begin(), vec.end(), std::greater<>());
    for (auto &p : vec) selected.push_back(p.second);
    return selected;
}

int count_free_vars(int M) {
    int cnt = 0;
    for (int v = 1; v <= n_vars && v <= M; ++v) {
        if (bimp_size[lit_index(v)] > 0 || bimp_size[lit_index(-v)] > 0) cnt++;
    }
    return cnt;
}

int main(int argc, char** argv) {
    auto total_start = std::chrono::high_resolution_clock::now();
    Options opt = parse_args(argc, argv);
    if (opt.filename.empty()) {
        printf("CNF file not specified\n");
        return 1;
    }
    parse_cnf(opt.filename.c_str());

    printf("%d variables will be considered for cubing\n", opt.m_vars);
    printf("No. of free variables: %d\n", count_free_vars(opt.m_vars));

    // Preselect variables using the heuristic
    std::vector<int> vars = preselect_vars(opt.m_vars);

    // Simple DFS based cube generation
    auto cubing_start = std::chrono::high_resolution_clock::now();
    std::vector<Cube> cubes;
    std::vector<int> path;
    int count = 0;
    for (int i = 0; i < opt.num_sims; ++i) {
        dfs(path, 0, count, opt, vars, cubes);
    }
    auto cubing_end = std::chrono::high_resolution_clock::now();

    if (!opt.out_file.empty()) {
        std::ofstream out(opt.out_file);
        for (auto &c : cubes) {
            for (int lit : c.lits) out << lit << " ";
            out << "0\n";
        }
        printf("Saved cubes to file  %s\n", opt.out_file.c_str());
    }

    double cubing_time = std::chrono::duration<double>(cubing_end - cubing_start).count();
    printf("Time taken for cubing:  %.3f\n", cubing_time);
    printf("Number of nodes:  %d\n", node_count);

    double total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();
    printf("Tool runtime:  %.3f\n", total_time);

    printf("Generated %zu cubes\n", cubes.size());
    return 0;
}

