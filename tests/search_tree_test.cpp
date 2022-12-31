// Copyright 2022 Pierre Talbot

#include "search_tree.hpp"
#include "helper.hpp"

using SplitInputLB = Split<IStore, InputOrder<IStore>, LowerBound<IStore>>;
using ST = SearchTree<IStore, SplitInputLB>;

template <class A>
void check_solution(A& a, vector<int> solution) {
  for(int i = 0; i < solution.size(); ++i) {
    EXPECT_EQ(a.project(AVar(sty, i)), Itv(solution[i]));
  }
}

TEST(SearchTreeTest, EnumerationSolution) {
  auto store = make_shared<IStore, StandardAllocator>(sty, 3);
  auto split = make_shared<SplitInputLB, StandardAllocator>(SplitInputLB(split_ty, store, store));
  auto search_tree = ST(tty, store, split);
  EXPECT_TRUE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  VarEnv<StandardAllocator> env;
  interpret_and_tell(search_tree,
    "var int: x1; var int: x2; var int: x3;\
    constraint int_ge(x1, 0); constraint int_le(x1, 2);\
    constraint int_ge(x2, 0); constraint int_le(x2, 2);\
    constraint int_ge(x3, 0); constraint int_le(x3, 2);", env);

  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  AbstractDeps<StandardAllocator> deps;
  ST sol(search_tree, deps);

  int solutions = 0;
  for(int x1 = 0; x1 < 3; ++x1) {
    for(int x2 = 0; x2 < 3; ++x2) {
      for(int x3 = 0; x3 < 3; ++x3) {
        // Going down a branch of the search tree.
        bool leaf = false;
        while(!leaf) {
          local::BInc has_changed;
          split->reset();
          GaussSeidelIteration::iterate(*split, has_changed);
          // There is no constraint so we are always navigating the under-approximated space.
          EXPECT_TRUE(search_tree.extract(sol));
          // All variables are supposed to be assigned if nothing changed.
          if(!has_changed.value()) {
            check_solution(sol, {x1, x2, x3});
            solutions++;
            leaf = true;
          }
          search_tree.refine(env, has_changed);
          EXPECT_TRUE(has_changed);
        }
      }
    }
  }
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  local::BInc has_changed;
  search_tree.refine(env, has_changed);
  EXPECT_FALSE(has_changed);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_EQ(solutions, 3*3*3);
}

using ISplitInputLB = Split<IPC, InputOrder<IPC>, LowerBound<IPC>>;
using IST = SearchTree<IPC, ISplitInputLB>;

TEST(SearchTreeTest, ConstrainedEnumeration) {
  auto store = make_shared<IStore, StandardAllocator>(sty, 3);
  auto ipc = make_shared<IPC, StandardAllocator>(IPC(pty, store));
  auto split = make_shared<ISplitInputLB, StandardAllocator>(ISplitInputLB(split_ty, ipc, ipc));
  auto search_tree = IST(tty, ipc, split);
  EXPECT_TRUE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  VarEnv<StandardAllocator> env;
  interpret_and_tell(search_tree,
    "var int: x1; var int: x2; var int: x3;\
    constraint int_ge(x1, 0); constraint int_le(x1, 2);\
    constraint int_ge(x2, 0); constraint int_le(x2, 2);\
    constraint int_ge(x3, 0); constraint int_le(x3, 2);\
    constraint int_plus(x1, x2, x3);", env);

  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  AbstractDeps<StandardAllocator> deps;
  IST sol(search_tree, deps);

  int solutions = 0;
  vector<vector<int>> sols = {
    {0, 0, 0},
    {0, 1, 1},
    {0, 2, 2},
    {1, 0, 1},
    {1, 1, 2},
    {2, 0, 2}} ;
  local::BInc has_changed = true;
  int iterations = 0;
  while(has_changed) {
    ++iterations;
    has_changed = false;
    GaussSeidelIteration::fixpoint(*ipc, has_changed);
    split->reset();
    GaussSeidelIteration::iterate(*split, has_changed);
    if(search_tree.extract(sol)) {
      check_solution(sol, sols[solutions++]);
    }
    search_tree.refine(env, has_changed);
  }
  EXPECT_EQ(iterations, 12);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  has_changed = local::BInc::bot();
  GaussSeidelIteration::fixpoint(*ipc, has_changed);
  search_tree.refine(env, has_changed);
  EXPECT_FALSE(has_changed);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_EQ(solutions, sols.size());
}
