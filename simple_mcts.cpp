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
    bool debug = false;
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
        } else if (arg == "-debug") {
            opt.debug = true;
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
    struct Node {
        std::vector<int> chosen_vars;
        std::vector<int> remaining_vars;
        std::unordered_map<int, std::unique_ptr<Node>> child; // chosen var -> child node
        std::unordered_map<int, double> Q; // action-value per variable
        std::unordered_map<int, int> N;    // visit count per variable
        int depth = 0;

        Node(std::vector<int> chosen, std::vector<int> remaining, int d)
            : chosen_vars(std::move(chosen)), remaining_vars(std::move(remaining)), depth(d) {}
    };

    const Options& opt;
    const std::unordered_map<int, std::pair<double,double>>& split_reward; // var -> (pos, neg)
    double cpuct = 1.4;
    std::unique_ptr<Node> root;
    int node_created = 0;

    MCTS(const std::vector<int>& ranked_vars,
         const Options& o,
         const std::unordered_map<int, std::pair<double,double>>& rewards)
        : opt(o), split_reward(rewards) {
        root = std::make_unique<Node>(std::vector<int>{}, ranked_vars, 0);
        node_created = 1;
    }

    bool is_terminal(const Node* node) const {
        if (opt.d_cutoff != -1 && node->depth >= opt.d_cutoff) return true;
        if (opt.n_cutoff != -1 && (int)node->chosen_vars.size() >= opt.n_cutoff) return true;
        if (node->remaining_vars.empty()) return true;
        return false;
    }

    static std::vector<int> remove_var(const std::vector<int>& xs, int v) {
        std::vector<int> out;
        out.reserve(xs.size());
        for (int x : xs) if (x != v) out.push_back(x);
        return out;
    }

    // Forest-style split reward for variable v: split on -v and +v, average both branches.
    double action_reward(int v) const {
        auto it = split_reward.find(v);
        if (it == split_reward.end()) return 0.0;
        return (it->second.first + it->second.second) / 2.0;
    }

    std::vector<std::tuple<double,int,double,int>> rank_actions_by_uct(Node* node) {
        std::vector<std::tuple<double,int,double,int>> scored; // uct, var, avg_reward, visits
        scored.reserve(node->remaining_vars.size());

        int totalN = 0;
        for (int v : node->remaining_vars) totalN += node->N[v];

        for (int v : node->remaining_vars) {
            double q = node->Q[v];
            int n = node->N[v];
            double explore = cpuct * sqrt((double)(totalN + 1e-6)) / (1 + n);
            double u = q + explore;
            scored.push_back({u, v, action_reward(v), n});
        }

        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
            return std::get<1>(a) < std::get<1>(b);
        });
        return scored;
    }

    double search(Node* node, int sim_id, int trace_depth = 0) {
        if (is_terminal(node)) {
            if (opt.debug) {
                printf("[sim %d][depth %d] terminal chosen_vars=%zu value=0.000\n",
                       sim_id, trace_depth, node->chosen_vars.size());
            }
            return 0.0;
        }

        auto scored = rank_actions_by_uct(node);
        if (opt.debug) {
            printf("[sim %d][depth %d] top actions by UCT (up to 3):\n", sim_id, trace_depth);
            for (size_t i = 0; i < scored.size() && i < 3; ++i) {
                printf("  #%zu var=%d UCT=%.6f avg_split_rew=%.6f Q=%.6f N=%d\n",
                       i + 1,
                       std::get<1>(scored[i]),
                       std::get<0>(scored[i]),
                       std::get<2>(scored[i]),
                       node->Q[std::get<1>(scored[i])],
                       std::get<3>(scored[i]));
            }
        }

        int best_v = std::get<1>(scored[0]);
        auto rewards = split_reward.at(best_v);
        double split_avg = (rewards.first + rewards.second) / 2.0;

        if (opt.debug) {
            printf("[sim %d][depth %d] choose var=%d, split rewards: -%d=>%.6f +%d=>%.6f avg=%.6f\n",
                   sim_id, trace_depth, best_v, best_v, rewards.first, best_v, rewards.second, split_avg);
        }

        if (!node->child.count(best_v)) {
            auto next_chosen = node->chosen_vars;
            next_chosen.push_back(best_v);
            auto next_remaining = remove_var(node->remaining_vars, best_v);
            node->child[best_v] = std::make_unique<Node>(std::move(next_chosen), std::move(next_remaining), node->depth + 1);
            node_created++;
            if (opt.debug) {
                printf("[sim %d][depth %d] expand child for var=%d -> child_depth=%d\n",
                       sim_id, trace_depth, best_v, node->depth + 1);
            }
        }

        double child_value = search(node->child[best_v].get(), sim_id, trace_depth + 1);
        double v = (split_avg + child_value) / 2.0;

        node->N[best_v]++;
        node->Q[best_v] += (v - node->Q[best_v]) / node->N[best_v];

        if (opt.debug) {
            printf("[sim %d][depth %d] backprop var=%d newQ=%.6f newN=%d value=%.6f\n",
                   sim_id, trace_depth, best_v, node->Q[best_v], node->N[best_v], v);
        }
        return v;
    }

    std::vector<int> run() {
        for (int i = 0; i < opt.num_sims; ++i) {
            if (opt.debug) printf("=== MCTS simulation %d/%d ===\n", i + 1, opt.num_sims);
            double val = search(root.get(), i + 1);
            if (opt.debug) {
                int total = 0;
                for (int v : root->remaining_vars) total += root->N[v];
                printf("[sim %d] finished value=%.6f root_total_visits=%d\n", i + 1, val, total);
            }
        }

        std::vector<int> best_vars;
        Node* node = root.get();
        while (!is_terminal(node)) {
            int best_v = -1;
            int best_n = -1;
            for (int v : node->remaining_vars) {
                int n = node->N[v];
                if (n > best_n) {
                    best_n = n;
                    best_v = v;
                }
            }
            if (best_v == -1) break;
            best_vars.push_back(best_v);
            if (!node->child.count(best_v)) break;
            node = node->child[best_v].get();
        }

        if (opt.debug) {
            printf("Final best variable sequence by visit count:");
            for (int v : best_vars) printf(" %d", v);
            printf("\n");
        }
        return best_vars;
    }
};

// Score only "free" (preselected) variables using propagation.
struct VarScore {
    int var = 0;
    int pos = 0;
    int neg = 0;
    double raw_score = 0.0;
    double norm_pos = 0.0;
    double norm_neg = 0.0;
    double norm_raw = 0.0;
};

std::vector<VarScore> preselect_vars(int M) {
    std::vector<VarScore> ranked;

    for (int v = 1; v <= n_vars && v <= M; v++) {
        if (!is_preselected(v)) continue;
        std::vector<int> propagated;
        int pos = 0, neg = 0;

        reset_assignments();
        if (unit_propagation(v, propagated)) pos = propagated.size();

        reset_assignments();
        if (unit_propagation(-v, propagated)) neg = propagated.size();

        double raw = (double)pos * (double)neg + 10.0 * (pos + neg) +
                     100.0 * (bimp_size[lit_index(v)] + bimp_size[lit_index(-v)]) +
                     5.0 * var_activity[v];

        ranked.push_back({v, pos, neg, raw, 0.0, 0.0, 0.0});
    }

    int max_prop = 1;
    double max_raw = 1.0;
    for (const auto& s : ranked) {
        max_prop = std::max(max_prop, std::max(s.pos, s.neg));
        max_raw = std::max(max_raw, s.raw_score);
    }

    for (auto& s : ranked) {
        s.norm_pos = (double)s.pos / (double)max_prop;
        s.norm_neg = (double)s.neg / (double)max_prop;
        s.norm_raw = s.raw_score / max_raw;
    }

    std::sort(ranked.begin(), ranked.end(), [](const VarScore &a, const VarScore &b) {
        return a.norm_raw > b.norm_raw;
    });
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
    if (opt.debug) {
        printf("Free variables:");
        for (int v : free_vars) printf(" %d", v);
        printf("\n");
    }

    auto score_start = std::chrono::high_resolution_clock::now();
    auto ranked = preselect_vars(opt.m_vars);
    auto score_end = std::chrono::high_resolution_clock::now();

    std::vector<int> vars;
    std::unordered_map<int, std::pair<double,double>> split_reward;
    for (const auto& s : ranked) {
        vars.push_back(s.var);
        split_reward[s.var] = {s.norm_neg, s.norm_pos}; // (-v, +v)
    }

    if (opt.debug) {
        printf("Variable ranking with normalization (var:raw:norm_raw:pos:neg:norm_pos:norm_neg):\n");
        for (size_t i = 0; i < ranked.size(); ++i) {
            const auto& s = ranked[i];
            printf("%zu. %d:%.3f:%.6f:%d:%d:%.6f:%.6f\n",
                   i + 1, s.var, s.raw_score, s.norm_raw, s.pos, s.neg, s.norm_pos, s.norm_neg);
        }
    }

    auto mcts_start = std::chrono::high_resolution_clock::now();
    MCTS mcts(vars, opt, split_reward);
    auto best_vars = mcts.run();
    auto mcts_end = std::chrono::high_resolution_clock::now();

    auto cube_gen_start = std::chrono::high_resolution_clock::now();
    std::vector<Cube> cubes = generate_cubes(best_vars);
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
    double parse_time = std::chrono::duration<double>(io_end - io_start).count();
    double score_time = std::chrono::duration<double>(score_end - score_start).count();
    double mcts_time = std::chrono::duration<double>(mcts_end - mcts_start).count();
    double cube_time = std::chrono::duration<double>(cube_gen_end - cube_gen_start).count();
    double write_time = std::chrono::duration<double>(write_end - write_start).count();
    if (opt.debug) {
        printf("Parsing time: %.3f\n", parse_time);
        printf("Scoring time: %.3f\n", score_time);
        printf("MCTS time: %.3f\n", mcts_time);
        printf("Cube gen time: %.3f\n", cube_time);
        printf("Write time: %.3f\n", write_time);
        double accounted = parse_time + score_time + mcts_time + cube_time + write_time;
        printf("Accounted component time: %.3f\n", accounted);
    }

    double cubing_time = score_time + mcts_time + cube_time + write_time;
    printf("Time taken for cubing:  %.3f\n", cubing_time);

    printf("Number of nodes:  %d\n", mcts.node_created);
    double total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();
    printf("Tool runtime:  %.3f\n", total_time);
    if (opt.debug) printf("Generated %zu cubes\n", cubes.size());
    return 0;
}
