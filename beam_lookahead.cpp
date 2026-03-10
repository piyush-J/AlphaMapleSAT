#include <bits/stdc++.h>
using namespace std;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

#define MAX_VARS 20000
#define MAX_CLAUSES 1000000
#define BIMP_GROW_CHUNK 4
#define ASSIGN_NONE 0
#define ASSIGN_TRUE 1
#define ASSIGN_FALSE 2

int n_vars, n_clauses;
int *clauses[MAX_CLAUSES];
int clause_sizes[MAX_CLAUSES];
int watched[MAX_CLAUSES][2];  // Two watched literal indices per clause

uint8_t assignments[MAX_VARS + 1];
int current_stamp = 1;
uint8_t is_unit_var[MAX_VARS + 1];

int *bimp[MAX_VARS * 2 + 2];
int bimp_size[MAX_VARS * 2 + 2];
int bimp_capacity[MAX_VARS * 2 + 2];
int global_queue[MAX_VARS + 1];

// Watch lists: for each literal, list of clause indices watching it
vector<int> watch_list[MAX_VARS * 2 + 2];

static inline int lit_index(int lit) {
    return (lit > 0) ? lit : n_vars - lit;
}

static inline bool is_lit_true(int lit) {
    int var = abs(lit);
    if (assignments[var] == ASSIGN_NONE) return false;
    return (lit > 0 && assignments[var] == ASSIGN_TRUE) ||
           (lit < 0 && assignments[var] == ASSIGN_FALSE);
}

static inline bool is_lit_false(int lit) {
    int var = abs(lit);
    if (assignments[var] == ASSIGN_NONE) return false;
    return (lit > 0 && assignments[var] == ASSIGN_FALSE) ||
           (lit < 0 && assignments[var] == ASSIGN_TRUE);
}

static inline void reset_assignments() {
    memset(assignments, ASSIGN_NONE, sizeof(uint8_t) * (n_vars + 1));
    current_stamp++;
}

static inline bool is_preselected(int var) {
    if (is_unit_var[var]) return false;  // Exclude unit clause variables
    return bimp_size[lit_index(var)] > 0 || bimp_size[lit_index(-var)] > 0;
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

    memset(is_unit_var, 0, sizeof(is_unit_var));
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

            clauses[clause_index] = (int*)malloc(sizeof(int) * size);
            memcpy(clauses[clause_index], lits, sizeof(int) * size);
            clause_sizes[clause_index] = size;

            if (size == 1) {
                is_unit_var[abs(lits[0])] = 1;
            }
            if (size == 2) {
                add_bimp(-lits[0], lits[1]);
                add_bimp(-lits[1], lits[0]);
            }
            
            // Initialize watched literals for clauses with size > 2
            if (size > 2) {
                watched[clause_index][0] = 0;
                watched[clause_index][1] = 1;
                watch_list[lit_index(lits[0])].push_back(clause_index);
                watch_list[lit_index(lits[1])].push_back(clause_index);
            }
            
            clause_index++;
        }
    }
    n_clauses = clause_index;
    fclose(fp);
}

int propagate_big_clauses(vector<int>& propagated) {
    // For 2-watched literals, we don't iterate through all clauses
    // Instead, we process the watch lists when literals become false
    // This is handled in the main propagation loop
    return 1;
}

int update_watches_and_propagate(int false_lit, vector<int>& propagated, int& queue_rear) {
    int lit_idx = lit_index(false_lit);
    vector<int>& watches = watch_list[lit_idx];
    
    // Process all clauses watching this literal
    for (size_t i = 0; i < watches.size(); ) {
        int clause_idx = watches[i];
        int *lits = clauses[clause_idx];
        int size = clause_sizes[clause_idx];
        
        if (size <= 2) {
            i++;
            continue;
        }
        
        // Find which watch position contains the false literal
        int watch_pos = -1;
        if (lits[watched[clause_idx][0]] == false_lit) {
            watch_pos = 0;
        } else if (lits[watched[clause_idx][1]] == false_lit) {
            watch_pos = 1;
        } else {
            // This literal is no longer watched, skip
            i++;
            continue;
        }
        
        int other_watch_pos = 1 - watch_pos;
        int other_watch_lit = lits[watched[clause_idx][other_watch_pos]];
        
        // Check if other watch is satisfied
        if (is_lit_true(other_watch_lit)) {
            i++;
            continue;
        }
        
        // Try to find a new literal to watch
        bool found_new_watch = false;
        for (int j = 0; j < size; j++) {
            if (j == watched[clause_idx][0] || j == watched[clause_idx][1]) continue;
            
            int lit = lits[j];
            if (!is_lit_false(lit)) {
                // Found a new literal to watch
                // Remove from current watch list
                watches[i] = watches.back();
                watches.pop_back();
                
                // Update watch and add to new watch list
                watched[clause_idx][watch_pos] = j;
                watch_list[lit_index(lit)].push_back(clause_idx);
                found_new_watch = true;
                break;
            }
        }
        
        if (!found_new_watch) {
            // Could not find new watch, check clause status
            if (is_lit_false(other_watch_lit)) {
                // Both watches are false - conflict
                return 0;
            } else {
                // Other watch must be unassigned - unit clause
                int var = abs(other_watch_lit);
                if (assignments[var] == ASSIGN_NONE) {
                    assignments[var] = (other_watch_lit > 0) ? ASSIGN_TRUE : ASSIGN_FALSE;
                    propagated.push_back(var);
                    global_queue[queue_rear++] = other_watch_lit;
                }
            }
            i++;
        }
        // Note: if found_new_watch is true, we don't increment i 
        // because we've already swapped with the last element
    }
    
    return 1;
}

int propagate_with_assumptions(const vector<int>& assumptions, vector<int>& propagated) {
    reset_assignments();
    propagated.clear();

    int front = 0, rear = 0;
    
    // Add all assumptions to the queue
    for (int lit : assumptions) {
        int var = abs(lit);
        if (assignments[var] != ASSIGN_NONE) {
            if ((assignments[var] == ASSIGN_TRUE && lit < 0) ||
                (assignments[var] == ASSIGN_FALSE && lit > 0)) {
                return 0;
            }
            continue;
        }
        assignments[var] = (lit > 0) ? ASSIGN_TRUE : ASSIGN_FALSE;
        propagated.push_back(var);
        global_queue[rear++] = lit;
    }

    // Main propagation loop
    while (front < rear) {
        int lit = global_queue[front++];
        
        // Propagate binary implications
        int idx = lit_index(lit);
        for (int i = 0; i < bimp_size[idx]; i++) {
            int implied = bimp[idx][i];
            int var = abs(implied);
            
            if (assignments[var] != ASSIGN_NONE) {
                if ((assignments[var] == ASSIGN_TRUE && implied < 0) ||
                    (assignments[var] == ASSIGN_FALSE && implied > 0)) {
                    return 0;
                }
                continue;
            }
            
            assignments[var] = (implied > 0) ? ASSIGN_TRUE : ASSIGN_FALSE;
            propagated.push_back(var);
            global_queue[rear++] = implied;
        }
        
        // Update watches for the negation of the assigned literal
        int false_lit = -lit;
        if (!update_watches_and_propagate(false_lit, propagated, rear)) {
            return 0;
        }
    }
    
    return 1;
}

int formula_score(const vector<int>& assumptions) {
    vector<int> propagated;
    if (!propagate_with_assumptions(assumptions, propagated)) return 0;
    return (int)propagated.size();
}

struct VarScore {
    int var = 0;
    int pos = 0;
    int neg = 0;
    double score = 0.0;
    
    VarScore() {}
    VarScore(int v, int p, int n, double s) : var(v), pos(p), neg(n), score(s) {}
};

VarScore score_variable_under_base(int var, const vector<int>& base_assumptions) {
    vector<int> plus = base_assumptions;
    plus.push_back(var);
    int pos = formula_score(plus);

    vector<int> minus = base_assumptions;
    minus.push_back(-var);
    int neg = formula_score(minus);

    double score = (double)pos * (double)neg + (double)pos + (double)neg;
    return VarScore(var, pos, neg, score);
}

vector<VarScore> rank_all_vars(int max_var, const vector<int>& base_assumptions) {
    vector<VarScore> out;
    for (int v = 1; v <= max_var; ++v) {
        if (!is_preselected(v)) continue;
        out.push_back(score_variable_under_base(v, base_assumptions));
    }
    sort(out.begin(), out.end(), [](const VarScore& a, const VarScore& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.var < b.var;
    });
    return out;
}

struct UpdatedSeedScore {
    int seed_var = 0;
    double s1 = 0.0;
    double s2_balanced = 0.0;
    double s1_norm = 0.0;
    double s2_norm = 0.0;
    double updated_score = 0.0;
    
    UpdatedSeedScore() {}
    UpdatedSeedScore(int v, double s1_val, double s2_bal, double s1n, double s2n, double upd)
        : seed_var(v), s1(s1_val), s2_balanced(s2_bal), s1_norm(s1n), s2_norm(s2n), updated_score(upd) {}
};

int main(int argc, char** argv) {
    auto total_start = std::chrono::high_resolution_clock::now();
    bool debug = false;
    bool lightweight = false;
    string filename;
    string out_file;
    int m = -1;

    if (argc < 2) {
        printf("Usage: %s <cnf-file> -o <cube-file> [-m M] [-debug] [--lightweight]\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            out_file = argv[++i];
        } else if (arg == "-debug") {
            debug = true;
        } else if (arg == "--lightweight") {
            lightweight = true;
        } else if (!arg.empty() && arg[0] != '-') {
            filename = arg;
        }
    }

    if (filename.empty() || out_file.empty()) {
        printf("Usage: %s <cnf-file> -o <cube-file> [-m M] [-debug] [--lightweight]\n", argv[0]);
        return 1;
    }

    auto parse_start = std::chrono::high_resolution_clock::now();
    parse_cnf(filename.c_str());
    auto parse_end = std::chrono::high_resolution_clock::now();
    if (m == -1 || m > n_vars) m = n_vars;

    // Count and list free variables
    vector<int> free_vars;
    for (int v = 1; v <= m; ++v) {
        if (is_preselected(v)) {
            free_vars.push_back(v);
        }
    }
    printf("Number of free variables in first %d vars: %zu\n", m, free_vars.size());
    if (debug) {
        printf("[debug] Free variables (vars with binary implications):");
        for (int v : free_vars) printf(" %d", v);
        printf("\n");
    }

    const double lambda = 0.5;
    const double gamma = 0.2;

    printf("Beam-lookahead scoring on first %d variables\n", m);
    if (debug) {
        printf("[debug] n_vars=%d n_clauses=%d m=%d\n", n_vars, n_clauses, m);
        printf("[debug] score(var|base)=pos*neg + pos + neg\n");
        printf("[debug] depth-1 S1(x)=score(x|base=[])\n");
        printf("[debug] depth-2 S2+(x)=score(x_j+|base=[x]), S2-(x)=score(x_j-|base=[-x])\n");
        printf("[debug] risk-aware S2(x)=mean(S2+,S2-) - gamma*|S2+-S2-|, gamma=%.2f\n", gamma);
        printf("[debug] normalized combine: updated=(1-lambda)*S1_norm + lambda*S2_norm, lambda=%.2f\n", lambda);
    }

    auto score_start = std::chrono::high_resolution_clock::now();
    auto ranked = rank_all_vars(m, {});
    auto score_end = std::chrono::high_resolution_clock::now();
    vector<int> top3;
    for (int i = 0; i < 3 && i < (int)ranked.size(); ++i) top3.push_back(ranked[i].var);

    printf("Top-%zu variables on original formula:\n", top3.size());
    for (size_t i = 0; i < top3.size(); ++i) {
        const auto& s = ranked[i];
        printf("%zu) var=%d pos=%d neg=%d score=%.3f\n", i + 1, s.var, s.pos, s.neg, s.score);
        if (debug) {
            printf("   [debug] calc: %d*%d + %d + %d = %.3f\n", s.pos, s.neg, s.pos, s.neg, s.score);
        }
    }

    int best_x;
    double best_score;
    auto lookahead_start = std::chrono::high_resolution_clock::now();
    
    if (lightweight) {
        // Lightweight mode: use depth-1 scoring only
        if (ranked.empty()) {
            printf("No variable candidates found.\n");
            return 1;
        }
        best_x = ranked[0].var;
        best_score = ranked[0].score;
        auto lookahead_end = std::chrono::high_resolution_clock::now();
        
        printf("\n[Lightweight mode] Best variable from depth-1 scoring: x=%d score=%.3f\n", best_x, best_score);
        if (debug) {
            printf("[debug] Skipped depth-2 lookahead in lightweight mode\n");
        }
        
        auto write_start = std::chrono::high_resolution_clock::now();
        ofstream out(out_file);
        out << "a " << -best_x << " 0\n";
        out << "a " << best_x << " 0\n";
        out.close();
        auto write_end = std::chrono::high_resolution_clock::now();
        printf("Saved cubes to file  %s\n", out_file.c_str());

        double parse_time = std::chrono::duration<double>(parse_end - parse_start).count();
        double score_time = std::chrono::duration<double>(score_end - score_start).count();
        double lookahead_time = std::chrono::duration<double>(lookahead_end - lookahead_start).count();
        double write_time = std::chrono::duration<double>(write_end - write_start).count();
        double total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();

        printf("Parse time: %.3f\n", parse_time);
        printf("Initial scoring time: %.3f\n", score_time);
        printf("Lookahead time: %.3f\n", lookahead_time);
        printf("Write time: %.3f\n", write_time);
        printf("Total runtime: %.3f\n", total_time);
        
        return 0;
    }
    
    // Full beam lookahead mode
    vector<UpdatedSeedScore> updated;
    for (int x : top3) {
        VarScore s1 = score_variable_under_base(x, {});
        double s2_plus = 0.0, s2_minus = 0.0;

        for (int sign : {+1, -1}) {
            int lit = sign * x;
            vector<int> base = {lit};
            auto rescored = rank_all_vars(m, base);

            int partner = x;
            for (const auto& rs : rescored) {
                if (rs.var != x) {
                    partner = rs.var;
                    break;
                }
            }

            VarScore partner_score = score_variable_under_base(partner, {lit});
            if (sign > 0) s2_plus = partner_score.score;
            else s2_minus = partner_score.score;

            if (debug) {
                printf("\n[debug] seed x=%d, branch lit=%d\n", x, lit);
                printf("        top-3 vars in subformula F AND %d:\n", lit);
                for (int i = 0; i < 3 && i < (int)rescored.size(); ++i) {
                    const auto& r = rescored[i];
                    printf("          #%d var=%d pos=%d neg=%d score=%.3f\n", i + 1, r.var, r.pos, r.neg, r.score);
                }
                printf("        chosen partner x_j=%d\n", partner);
                printf("        S2_branch uses score(x_j | base=[%d])\n", lit);
                printf("        x_j details: pos=%d neg=%d score=%.3f\n", partner_score.pos, partner_score.neg, partner_score.score);
                printf("        calc: %d*%d + %d + %d = %.3f\n",
                       partner_score.pos, partner_score.neg, partner_score.pos, partner_score.neg, partner_score.score);
            }
        }

        double s2_mean = (s2_plus + s2_minus) / 2.0;
        double s2_balanced = s2_mean - gamma * fabs(s2_plus - s2_minus);

        updated.push_back(UpdatedSeedScore(x, s1.score, s2_balanced, 0.0, 0.0, 0.0));

        if (debug) {
            printf("\n[debug] depth summary for x=%d\n", x);
            printf("        S1=%.3f\n", s1.score);
            printf("        S2+=%.3f, S2-=%.3f\n", s2_plus, s2_minus);
            printf("        S2_mean=(%.3f+%.3f)/2=%.3f\n", s2_plus, s2_minus, s2_mean);
            printf("        S2_balanced=%.3f - %.2f*|%.3f-%.3f| = %.3f\n",
                   s2_mean, gamma, s2_plus, s2_minus, s2_balanced);
        }
    }

    double max_s1 = 1e-9, max_s2 = 1e-9;
    for (const auto& u : updated) {
        max_s1 = max(max_s1, u.s1);
        max_s2 = max(max_s2, u.s2_balanced);
    }

    for (auto& u : updated) {
        u.s1_norm = u.s1 / max_s1;
        u.s2_norm = u.s2_balanced / max_s2;
        u.updated_score = (1.0 - lambda) * u.s1_norm + lambda * u.s2_norm;
        if (debug) {
            printf("\n[debug] normalize+combine for x=%d\n", u.seed_var);
            printf("        S1_norm=%.3f/%.3f=%.6f\n", u.s1, max_s1, u.s1_norm);
            printf("        S2_norm=%.3f/%.3f=%.6f\n", u.s2_balanced, max_s2, u.s2_norm);
            printf("        updated=(1-%.2f)*%.6f + %.2f*%.6f = %.6f\n",
                   lambda, u.s1_norm, lambda, u.s2_norm, u.updated_score);
        }
    }

    sort(updated.begin(), updated.end(), [](const UpdatedSeedScore& a, const UpdatedSeedScore& b) {
        if (a.updated_score != b.updated_score) return a.updated_score > b.updated_score;
        return a.seed_var < b.seed_var;
    });

    if (updated.empty()) {
        printf("No variable candidates found.\n");
        return 1;
    }

    auto lookahead_end = std::chrono::high_resolution_clock::now();

    best_x = updated[0].seed_var;
    best_score = updated[0].updated_score;
    printf("\nBest variable after beam lookahead: x=%d updated_score=%.6f\n", best_x, best_score);
    if (debug) {
        printf("[debug] final ranking among top-3 seeds:\n");
        for (size_t i = 0; i < updated.size(); ++i) {
            const auto& u = updated[i];
            printf("        #%zu x=%d S1=%.3f S2_bal=%.3f S1_norm=%.6f S2_norm=%.6f updated=%.6f\n",
                   i + 1, u.seed_var, u.s1, u.s2_balanced, u.s1_norm, u.s2_norm, u.updated_score);
        }
    }

    auto write_start = std::chrono::high_resolution_clock::now();
    ofstream out(out_file);
    out << "a " << -best_x << " 0\n";
    out << "a " << best_x << " 0\n";
    out.close();
    auto write_end = std::chrono::high_resolution_clock::now();
    printf("Saved cubes to file  %s\n", out_file.c_str());

    double parse_time = std::chrono::duration<double>(parse_end - parse_start).count();
    double score_time = std::chrono::duration<double>(score_end - score_start).count();
    double lookahead_time = std::chrono::duration<double>(lookahead_end - lookahead_start).count();
    double write_time = std::chrono::duration<double>(write_end - write_start).count();
    double total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();

    printf("Parse time: %.3f\n", parse_time);
    printf("Initial scoring time: %.3f\n", score_time);
    printf("Lookahead time: %.3f\n", lookahead_time);
    printf("Write time: %.3f\n", write_time);
    printf("Total runtime: %.3f\n", total_time);

    return 0;
}
