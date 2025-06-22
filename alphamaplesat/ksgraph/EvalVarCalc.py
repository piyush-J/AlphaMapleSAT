from pysat.solvers import Solver
from pysat.formula import CNF
import operator
from collections import defaultdict

class Node:

    def __init__(self, prior_actions=None, unsat_learnt_actions=None) -> None:
        if prior_actions is None:
            prior_actions = []
        if unsat_learnt_actions is None:
            unsat_learnt_actions = []
        self.prior_actions = prior_actions # list of literals
        self.unsat_learnt_actions = unsat_learnt_actions # list of literals whose negation leads to UNSAT in the next step
        self.reward = None # only for terminal nodes
        # self.best_var_rew = 0 # found via propagation on the parent node, the best variable has been added to prior_actions

        # found after propagating on the current node
        self.cutoff = False
        self.refuted = False
        self.next_best_var = None 
        # self.next_best_var_rew = None
        # self.all_var_rew = None

    def __str__(self) -> str:
        return f"Node: Prior actions: {self.prior_actions}, Unsat learnt actions: {self.unsat_learnt_actions}, Reward: {self.reward}, Cutoff: {self.cutoff}, Refuted: {self.refuted}, Next best var: {self.next_best_var}"

    def is_terminal(self):
        assert self.cutoff is not None and self.refuted is not None
        return self.cutoff or self.refuted
    
    def is_refuted(self):
        assert self.refuted is not None
        return self.refuted

    def get_next_best_var(self):
        return self.next_best_var

    def get_next_node(self, var):
        return Node(self.prior_actions+[var], self.unsat_learnt_actions[:])

    def valid_cubing_lits(self, literals_all, freevars_all): 
        negated_prior_actions = [-l for l in self.prior_actions]
        negated_unsat_learnt_actions = [-l for l in self.unsat_learnt_actions]
        # freevars = literals_all - non-free-vars
        return list(set(freevars_all) - set(self.prior_actions) - set(negated_prior_actions) - set(self.unsat_learnt_actions) - set(negated_unsat_learnt_actions))


class MarchPysatPropagate:

    def __init__(self, cnf, m) -> None:
        self.cnf = cnf
        self.nv = self.cnf.nv
        self.solver = Solver(name="minisat22", bootstrap_with=self.cnf)
        self.m = m # only top m variables to be considered for cubing

        if self.m is None: 
            self.m = self.cnf.nv
        print(f"{m} variables will be considered for cubing")

        BinaryImp = self.build_binary_implications(self.cnf.clauses)
        freevars = self.construct_freevars(BinaryImp)
        print(f"No. of free variables: {len(freevars)}")

        literals_pos = list(range(1, self.m+1))
        literals_neg = [-l for l in literals_pos]
        self.literals_all = literals_pos + literals_neg # we need this to always be the edge variables

        # self.non_free_variables = set(range(1, self.m+1)) - set(freevars)
        freevars_neg = [-l for l in freevars]
        self.freevars_all = freevars + freevars_neg

        self.node_count = 0
        self.cached_unsat_learnt_actions = {}

    def build_binary_implications(self, clauses):
        BinaryImp = defaultdict(list)
        for a, b in (clause for clause in clauses if len(clause) == 2):
            BinaryImp[-a].append(b)
            BinaryImp[-b].append(a)
        return BinaryImp

    def construct_freevars(self, BinaryImp):
        freevarsArray = [
            var for var in range(1, self.m + 1)
            if (len(BinaryImp[var]) > 0 or len(BinaryImp[-var]) > 0)  # has binary implications
        ]
        return freevarsArray

    def propagate(self, node):

        if tuple(node.prior_actions[:-1]) in self.cached_unsat_learnt_actions: # check if the parent node's unsat_learnt_actions are cached
            node.unsat_learnt_actions = self.cached_unsat_learnt_actions[tuple(node.prior_actions[:-1])][:]

        out1 = self.solver.propagate(assumptions=node.prior_actions+node.unsat_learnt_actions)
        assert out1 is not None
        not_unsat1, asgn1 = out1
        len_asgn_edge_vars = len(set(asgn1).intersection(set(self.literals_all))) # number of assigned edge variables

        # check for refutation
        if not not_unsat1: # on second thought, this should never happen because FLE is being used and len(all_lit_rew) == 0 will be caught before this
            assert False, "Refutation found in the parent node"
            node.refuted = True
            node.reward = 1.0 # max reward
            return 0, None, {k: 0 for k in range(1, self.m+1)} # pass an empty dict
        else:
            node.refuted = False

        #TODO: what if the result is SAT?

        while True:
            unsat_flag = False
            all_lit_rew = {}
            all_var_rew = {}
            valid_cubing_lits = node.valid_cubing_lits(self.literals_all, self.freevars_all)

            for literal in valid_cubing_lits:
                assert literal not in node.prior_actions+node.unsat_learnt_actions, "Duplicate literals in the list"
                out = self.solver.propagate(assumptions=node.prior_actions+node.unsat_learnt_actions+[literal])
                assert out is not None
                not_unsat2, asgn = out
                if not not_unsat2: # recompute if unsat and recomputation limit not reached
                    unsat_literal = literal
                    node.unsat_learnt_actions.append(-unsat_literal) # add the negation of the unsat literal to the learnt actions
                    unsat_flag = True
                    break

                all_lit_rew[literal] = len(set(asgn).intersection(set(self.literals_all)))

            if not unsat_flag: # no refutation found
                break
        
        if tuple(node.prior_actions) not in self.cached_unsat_learnt_actions:
            self.cached_unsat_learnt_actions[tuple(node.prior_actions)] = node.unsat_learnt_actions[:]

        if len(all_lit_rew) == 0:
            node.cutoff = True
            node.reward = 1.0 # no valid cubing literals found, so reward is max
            # print(f"Node: {node.prior_actions} - no valid cubing literals found, so reward is max")
            return 0, None, {k: 0 for k in range(1, self.m+1)}

        # combine the rewards of the positive and negative literals
        for literal in valid_cubing_lits:
            if literal > 0:
                all_var_rew[literal] = (all_lit_rew[literal] * all_lit_rew[-literal])/((len(node.prior_actions)+1)**2) + all_lit_rew[literal]/(len(node.prior_actions)+1) + all_lit_rew[-literal]/(len(node.prior_actions)+1)

        # get the key (var) of the best value (eval_var)
        next_best_var = max(all_var_rew.items(), key=operator.itemgetter(1))[0]

        return 1, len_asgn_edge_vars, all_var_rew