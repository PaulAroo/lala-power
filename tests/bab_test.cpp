// Copyright 2022 Pierre Talbot

#include "search_tree.hpp"
#include "bab.hpp"
#include "helper.hpp"

using SplitInputLB = Split<IStore, InputOrder<IStore>, LowerBound<IStore>>;
using ST = SearchTree<IStore, SplitInputLB>;
using BAB_ = BAB<ST>;

template <class A>
void check_solution(A& a, vector<Itv> solution) {
  for(int i = 0; i < solution.size(); ++i) {
    EXPECT_EQ(a.project(AVar(sty, i)), solution[i]);
  }
}

// Minimize is true, maximize is false.
void test_unconstrained_bab(bool mode) {
  auto store = make_shared<IStore, StandardAllocator>(sty, 3);
  auto split = make_shared<SplitInputLB, StandardAllocator>(SplitInputLB(split_ty, store, store));
  auto search_tree = make_shared<ST, StandardAllocator>(tty, store, split);
  auto bab = BAB_(bab_ty, search_tree);

  std::cout << "Try interpreted the constraint\n" << std::endl;
  VarEnv<StandardAllocator> env;
  interpret_and_tell(bab,
    ("var int: x1; var int: x2; var int: x3;\
    constraint int_ge(x1, 0); constraint int_le(x1, 2);\
    constraint int_ge(x2, 0); constraint int_le(x2, 2);\
    constraint int_ge(x3, 0); constraint int_le(x3, 2);\
    solve " + std::string(mode ? "minimize" : "maximize") + " x2;").c_str(), env);

std::cout << "Successfully interpreted the constraint\n" << std::endl;

  // Find solution optimizing x2.
  int iterations = 0;
  local::BInc has_changed = true;
  while(!bab.extract(bab) && has_changed) {
    iterations++;
    has_changed = false;
    split->reset();
    // Compute \f$ pop \circ push \circ split \circ bab \f$.
    bab.refine(env, has_changed);
    GaussSeidelIteration::iterate(*split, has_changed);
    search_tree->refine(env, has_changed);
  }
  // Find the optimum in the root node since they are no constraint...
  check_solution(bab.optimum(), {Itv(0,2),Itv(0,2),Itv(0,2)});
  // With a input-order smallest first strat, the fixed point is reached after 1 iteration.
  EXPECT_EQ(iterations, 1);

  EXPECT_TRUE(search_tree->is_top());

  // One more iteration to check idempotency.
  has_changed = false;
  split->reset();
  bab.refine(env, has_changed);
  GaussSeidelIteration::iterate(*split, has_changed);
  search_tree->refine(env, has_changed);
  EXPECT_FALSE(has_changed);
}

TEST(BABTest, UnconstrainedOptimization) {
  test_unconstrained_bab(true);
  test_unconstrained_bab(false);
}

using ISplitInputLB = Split<IPC, InputOrder<IPC>, LowerBound<IPC>>;
using IST = SearchTree<IPC, ISplitInputLB>;
using IBAB = BAB<IST>;

// Minimize is true, maximize is false.
void test_constrained_bab(bool mode) {
  auto store = make_shared<IStore, StandardAllocator>(sty, 3);
  auto ipc = make_shared<IPC, StandardAllocator>(IPC(pty, store));
  auto split = make_shared<ISplitInputLB, StandardAllocator>(ISplitInputLB(split_ty, ipc, ipc));
  auto search_tree = make_shared<IST, StandardAllocator>(tty, ipc, split);
  auto bab = IBAB(bab_ty, search_tree);

  // Interpret formula
  VarEnv<StandardAllocator> env;
  interpret_and_tell(bab,
    ("var int: x1; var int: x2; var int: x3;\
    constraint int_ge(x1, 0); constraint int_le(x1, 2);\
    constraint int_ge(x2, 0); constraint int_le(x2, 2);\
    constraint int_ge(x3, 0); constraint int_le(x3, 2);\
    constraint int_plus(x1, x2, x3);\
    solve " + std::string(mode ? "minimize" : "maximize") + " x3;").c_str(), env);

  // Find solution optimizing x3.
  local::BInc has_changed = true;
  int iterations = 0;
  while(!bab.extract(bab) && has_changed) {
    iterations++;
    has_changed = false;
    // Compute \f$ pop \circ push \circ split \circ bab \circ refine \f$.
    GaussSeidelIteration::fixpoint(*ipc, has_changed);
    bab.refine(env, has_changed);
    split->reset();
    GaussSeidelIteration::iterate(*split, has_changed);
    search_tree->refine(env, has_changed);
  }
  EXPECT_TRUE(bab.is_top());
  if(mode) {
    check_solution(bab.optimum(), {Itv(0,0),Itv(0,0),Itv(0,0)});
    EXPECT_EQ(iterations, 5);
  }
  else {
    check_solution(bab.optimum(), {Itv(0,0),Itv(2,2),Itv(2,2)});
    EXPECT_EQ(iterations, 7);
  }

  EXPECT_TRUE(search_tree->is_top());

  // One more iteration to check idempotency.
  has_changed = false;
  GaussSeidelIteration::fixpoint(*ipc, has_changed);
  bab.refine(env, has_changed);
  split->reset();
  GaussSeidelIteration::iterate(*split, has_changed);
  search_tree->refine(env, has_changed);
  EXPECT_FALSE(has_changed);
}

TEST(BABTest, ConstrainedOptimization) {
  test_constrained_bab(true);
  test_constrained_bab(false);
}
