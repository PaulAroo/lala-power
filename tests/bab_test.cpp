// Copyright 2022 Pierre Talbot

#include "lala/search_tree.hpp"
#include "lala/bab.hpp"
#include "helper.hpp"

using ST = SearchTree<IStore, SplitStrategy<IStore>>;
using BAB_ = BAB<ST, IStore>;

template <class A>
void check_solution(A& a, vector<Itv> solution) {
  for(int i = 0; i < solution.size(); ++i) {
    EXPECT_EQ(a.project(AVar(sty, i)), solution[i]);
  }
}

// Minimize is true, maximize is false.
void test_unconstrained_bab(bool mode) {
  SolverOutput<standard_allocator> output(standard_allocator{});
  lala::impl::FlatZincParser<standard_allocator> parser(output);
  auto f = parser.parse("array[1..3] of var 0..2: a;\
    solve::int_search(a, input_order, indomain_min, complete)" +
    std::string(mode ? "minimize" : "maximize") + " a[3];");
  EXPECT_TRUE(f);
  // f->print(false);
  VarEnv<standard_allocator> env;
  const size_t num_vars = 3;
  auto store = make_shared<IStore, standard_allocator>(env.extends_abstract_dom(), num_vars);
  auto split = make_shared<SplitStrategy<IStore>, standard_allocator>(env.extends_abstract_dom(), store);
  auto search_tree = make_shared<ST, standard_allocator>(env.extends_abstract_dom(), store, split);
  // Best is a copy of the store, therefore it must have the same abstract type (in particular, when projecting the variable).
  auto best = make_shared<IStore, standard_allocator>(store->aty(), num_vars);
  auto bab = BAB_(env.extends_abstract_dom(), search_tree, best);

  EXPECT_TRUE(bab.is_top());
  EXPECT_FALSE(bab.is_bot());

  IDiagnostics diagnostics;
  EXPECT_TRUE(interpret_and_tell<true>(*f, env, bab, diagnostics));

  EXPECT_FALSE(bab.is_bot());
  EXPECT_FALSE(bab.is_top());

  // Find solution optimizing a[3].
  bool has_changed{true};
  int iterations = 0;
  while(!bab.is_extractable() && has_changed) {
    iterations++;
    has_changed = false;
    // Compute \f$ pop \circ push \circ split \circ bab \f$.
    if(search_tree->is_extractable()) {
      has_changed |= bab.deduce();
    }
    has_changed |= search_tree->deduce();
  }
  // With a input-order smallest first strat, the fixed point is reached after 1 iteration.
  EXPECT_EQ(iterations, 1);
  // Find the optimum in the root node since they are no constraint...
  check_solution(*best, {Itv(0,2),Itv(0,2),Itv(0,2)});

  EXPECT_TRUE(search_tree->is_bot());

  // One more iteration to check idempotency.
  EXPECT_FALSE(search_tree->deduce());
}

TEST(BABTest, UnconstrainedOptimization) {
  test_unconstrained_bab(true);
  test_unconstrained_bab(false);
}

using IST = SearchTree<IPC, SplitStrategy<IPC>>;
using IBAB = BAB<IST, IStore>;

// Minimize is true, maximize is false.
void test_constrained_bab(bool mode) {
  SolverOutput<standard_allocator> output(standard_allocator{});
  lala::impl::FlatZincParser<standard_allocator> parser(output);
  auto f = parser.parse("array[1..3] of var 0..2: a;\
    constraint int_plus(a[1], a[2], a[3]);\
    solve::int_search(a, input_order, indomain_min, complete)" +
    std::string(mode ? "minimize" : "maximize") + " a[3];");
  EXPECT_TRUE(f);
  VarEnv<standard_allocator> env;
  const size_t num_vars = 3;
  auto store = make_shared<IStore, standard_allocator>(env.extends_abstract_dom(), num_vars);
  auto ipc = make_shared<IPC, standard_allocator>(IPC(env.extends_abstract_dom(), store));
  auto split = make_shared<SplitStrategy<IPC>, standard_allocator>(env.extends_abstract_dom(), ipc);
  auto search_tree = make_shared<IST, standard_allocator>(env.extends_abstract_dom(), ipc, split);
  // Best is a copy of the store, therefore it must have the same abstract type (in particular, when projecting the variable).
  auto best = make_shared<IStore, standard_allocator>(store->aty(), num_vars);
  auto bab = IBAB(env.extends_abstract_dom(), search_tree, best);

  using F = TFormula<standard_allocator>;
  IDiagnostics diagnostics;
  EXPECT_TRUE(interpret_and_tell<true>(*f, env, bab, diagnostics));

  // Find solution optimizing a[3].
  bool has_changed{true};
  int iterations = 0;
  while(!bab.is_extractable() && has_changed) {
    iterations++;
    has_changed = false;
    // Compute \f$ pop \circ push \circ split \circ bab \circ refine \f$.
    has_changed |= GaussSeidelIteration{}.fixpoint(*ipc);
    if(search_tree->is_extractable()) {
      has_changed |= bab.deduce();
    }
    has_changed |= search_tree->deduce();
  }
  EXPECT_TRUE(bab.is_bot());
  if(mode) {
    check_solution(*best, {Itv(0,0),Itv(0,0),Itv(0,0)});
    EXPECT_EQ(iterations, 5);
  }
  else {
    check_solution(bab.optimum(), {Itv(0,0),Itv(2,2),Itv(2,2)});
    EXPECT_EQ(iterations, 7);
  }

  EXPECT_TRUE(search_tree->is_bot());

  // One more iteration to check idempotency.
  has_changed = GaussSeidelIteration{}.fixpoint(*ipc);
  has_changed |= search_tree->deduce();
  EXPECT_FALSE(has_changed);
}

TEST(BABTest, ConstrainedOptimization) {
  test_constrained_bab(true);
  test_constrained_bab(false);
}
