#include <bits/stdc++.h>
using namespace std;

/**
 * Simplified C++ version of AlphaMapleSAT cubing.
 * This implementation focuses on argument parsing, CNF parsing,
 * a basic unit propagation based on the provided BCP routines,
 * and a Monte Carlo Tree Search (MCTS) routine that explores
 * variable assignments. The implementation is intentionally
 * lightweight but demonstrates the core ideas behind the
 * MCTS-based cubing strategy used in AlphaMapleSAT.
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

static inline bool is_preselected(int var) {
    return bimp_size[lit_index(var)] > 0 || bimp_size[lit_index(-var)] > 0;
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

// ------------------------------------------------------------
// Minimal MCTS implementation for exploring variable literals
// ------------------------------------------------------------

struct MCTSNode {
    std::vector<int> path;   // literals assigned so far
    MCTSNode* child[2] = {nullptr, nullptr};
    double Q[2] = {0.0, 0.0};
    int N[2] = {0, 0};
    bool terminal = false;
    int depth = 0;
    MCTSNode(const std::vector<int>& p, int d, bool term)
        : path(p), terminal(term), depth(d) {}
};

struct MCTS {
    const std::vector<int>& vars;
    const Options& opt;
    double cpuct = 1.4;
    MCTSNode* root;
    int node_created = 0;

    MCTS(const std::vector<int>& v, const Options& o)
        : vars(v), opt(o) {
        root = new MCTSNode({}, 0, is_terminal(0, {}));
        node_created = 1;
    }

    bool is_terminal(int depth, const std::vector<int>& path) const {
        if (opt.d_cutoff != -1 && depth >= opt.d_cutoff) return true;
        if (opt.n_cutoff != -1 && (int)path.size() >= opt.n_cutoff) return true;
        if (depth >= (int)vars.size()) return true;
        return false;
    }

    double rollout(MCTSNode* node) const {
        return (double)node->path.size();
    }

    void expand(MCTSNode* node) {
        if (node->terminal) return;
        int var = vars[node->depth];
        for (int a = 0; a < 2; ++a) {
            auto p = node->path;
            int lit = a ? var : -var;
            p.push_back(lit);
            bool term = is_terminal(node->depth + 1, p);
            node->child[a] = new MCTSNode(p, node->depth + 1, term);
            node_created++;
        }
    }

    double search(MCTSNode* node) {
        if (node->terminal) {
            return rollout(node);
        }
        if (node->child[0] == nullptr) {
            expand(node);
            return rollout(node);
        }

        int totalN = node->N[0] + node->N[1];
        double bestU = -1e9;
        int bestA = 0;
        for (int a = 0; a < 2; ++a) {
            double u = node->Q[a] + cpuct * sqrt((double)(totalN + 1e-6)) / (1 + node->N[a]);
            if (u > bestU) {
                bestU = u;
                bestA = a;
            }
        }
        double v = search(node->child[bestA]);
        node->N[bestA]++;
        node->Q[bestA] += (v - node->Q[bestA]) / node->N[bestA];
        return v;
    }

    std::vector<int> run() {
        for (int i = 0; i < opt.num_sims; ++i) search(root);
        std::vector<int> best;
        MCTSNode* node = root;
        while (!node->terminal) {
            int a = node->N[1] > node->N[0] ? 1 : 0;
            if (node->child[a] == nullptr) {
                int var = vars[node->depth];
                auto p = node->path;
                int lit_tmp = a ? var : -var;
                p.push_back(lit_tmp);
                bool term = is_terminal(node->depth + 1, p);
                node->child[a] = new MCTSNode(p, node->depth + 1, term);
                node_created++;
            }
            int lit = a ? vars[node->depth] : -vars[node->depth];
            best.push_back(lit);
            node = node->child[a];
        }
        return best;
    }
};

// Score variables using propagation as in the provided snippet
std::vector<std::pair<int,int>> preselect_vars(int M) {
    std::vector<std::pair<int,int>> ranked;
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

    for (int v = 1; v <= n_vars && v <= M; ++v) {
        if (scores[v][1] > 0) ranked.push_back({v, scores[v][1]});
    }
    std::sort(ranked.begin(), ranked.end(), [](auto &a, auto &b){return a.second > b.second;});
    return ranked;
}

int count_free_vars(int M) {
    int cnt = 0;
    for (int v = 1; v <= n_vars && v <= M; ++v) {
        if (is_preselected(v)) cnt++;
    }
    return cnt;
}

std::vector<int> list_free_vars(int M) {
    std::vector<int> lst;
    for (int v = 1; v <= n_vars && v <= M; ++v) {
        if (is_preselected(v)) lst.push_back(v);
    }
    return lst;
}

void enumerate_cubes_rec(const std::vector<int>& vars, int idx, std::vector<int>& cur,
                         std::vector<Cube>& cubes) {
    if (idx == (int)vars.size()) {
        cubes.push_back({cur});
        return;
    }
    int v = vars[idx];
    cur.push_back(-v);
    enumerate_cubes_rec(vars, idx + 1, cur, cubes);
    cur.back() = v;
    enumerate_cubes_rec(vars, idx + 1, cur, cubes);
    cur.pop_back();
}

std::vector<Cube> generate_cubes(const std::vector<int>& vars) {
    std::vector<Cube> cubes;
    std::vector<int> cur;
    enumerate_cubes_rec(vars, 0, cur, cubes);
    return cubes;
}

int main(int argc, char** argv) {
    auto total_start = std::chrono::high_resolution_clock::now();
    Options opt = parse_args(argc, argv);
    if (opt.filename.empty()) {
        printf("CNF file not specified\n");
        return 1;
    }

    auto io_start = std::chrono::high_resolution_clock::now();
    parse_cnf(opt.filename.c_str());
    auto io_end = std::chrono::high_resolution_clock::now();

    printf("%d variables will be considered for cubing\n", opt.m_vars);
    auto free_vars = list_free_vars(opt.m_vars);
    printf("No. of free variables: %zu\n", free_vars.size());
    printf("Free variables:");
    for (int v : free_vars) printf(" %d", v);
    printf("\n");

    auto score_start = std::chrono::high_resolution_clock::now();
    auto ranked = preselect_vars(opt.m_vars);
    auto score_end = std::chrono::high_resolution_clock::now();

    printf("Variable ranking (var:score):\n");
    for (size_t i = 0; i < ranked.size(); ++i) {
        printf("%zu. %d:%d\n", i+1, ranked[i].first, ranked[i].second);
    }

    std::vector<int> vars;
    for (auto &p : ranked) vars.push_back(p.first);

    auto mcts_start = std::chrono::high_resolution_clock::now();
    MCTS mcts(vars, opt);
    auto best_path = mcts.run();
    auto mcts_end = std::chrono::high_resolution_clock::now();

    std::vector<int> abs_vars;
    for (int lit : best_path) abs_vars.push_back(std::abs(lit));

    auto cube_gen_start = std::chrono::high_resolution_clock::now();
    std::vector<Cube> cubes = generate_cubes(abs_vars);
    auto cube_gen_end = std::chrono::high_resolution_clock::now();

    auto write_start = std::chrono::high_resolution_clock::now();
    if (!opt.out_file.empty()) {
        std::ofstream out(opt.out_file);
        for (auto &c : cubes) {
            out << "a";
            for (int lit : c.lits) out << " " << lit;
            out << " 0\n";
        }
        printf("Saved cubes to file  %s\n", opt.out_file.c_str());
    }
    auto write_end = std::chrono::high_resolution_clock::now();

    printf("Parsing time: %.3f\n", std::chrono::duration<double>(io_end - io_start).count());
    printf("Scoring time: %.3f\n", std::chrono::duration<double>(score_end - score_start).count());
    printf("MCTS time: %.3f\n", std::chrono::duration<double>(mcts_end - mcts_start).count());
    printf("Cube gen time: %.3f\n", std::chrono::duration<double>(cube_gen_end - cube_gen_start).count());
    printf("Write time: %.3f\n", std::chrono::duration<double>(write_end - write_start).count());

    printf("Number of nodes:  %d\n", mcts.node_created);
    double total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();
    printf("Tool runtime:  %.3f\n", total_time);

    printf("Generated %zu cubes\n", cubes.size());
    return 0;
}

