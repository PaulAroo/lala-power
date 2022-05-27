// Copyright 2022 Pierre Talbot

#include "search_tree.hpp"
#include "helper.hpp"

using SplitInputLB = Split<IStore, InputOrder<IStore>, LowerBound<IStore>>;
using ST = SearchTree<IStore, SplitInputLB>;

template <class A>
void check_solution(A& a, vector<int> solution) {
  for(int i = 0; i < solution.size(); ++i) {
    EXPECT_EQ2(a.project(make_var(sty, i)), Itv(solution[i]));
  }
}

TEST(SearchTreeTest, EnumerationSolution) {
  auto store = make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty)));
  populate_istore_n_vars(*store, 3, 0, 2);
  auto split = make_shared<SplitInputLB, StandardAllocator>(SplitInputLB(split_ty, store, store));
  auto search_tree = ST(tty, store, split);
  EXPECT_FALSE2(search_tree.is_bot());
  EXPECT_FALSE2(search_tree.is_top());

  AbstractDeps<StandardAllocator> deps;
  ST sol(search_tree, deps);

  int solutions = 0;
  for(int x0 = 0; x0 < 3; ++x0) {
    for(int x1 = 0; x1 < 3; ++x1) {
      for(int x2 = 0; x2 < 3; ++x2) {
        // Going down a branch of the search tree.
        bool leaf = false;
        while(!leaf) {
          BInc has_changed = BInc::bot();
          split->reset();
          seq_refine(*split, has_changed);
          // There is no constraint so we are always navigating the under-approximated space.
          EXPECT_TRUE(search_tree.extract(sol));
          // All variables are supposed to be assigned if nothing changed.
          if(!has_changed.value()) {
            check_solution(sol, {x0, x1, x2});
            solutions++;
            leaf = true;
          }
          search_tree.refine(has_changed);
          EXPECT_TRUE2(has_changed);
        }
      }
    }
  }
  EXPECT_TRUE2(search_tree.is_top());
  EXPECT_FALSE2(search_tree.is_bot());
  BInc has_changed = BInc::bot();
  search_tree.refine(has_changed);
  EXPECT_FALSE2(has_changed);
  EXPECT_TRUE2(search_tree.is_top());
  EXPECT_FALSE2(search_tree.is_bot());
  EXPECT_EQ(solutions, 3*3*3);
}

using ISplitInputLB = Split<IIPC, InputOrder<IIPC>, LowerBound<IIPC>>;
using IST = SearchTree<IIPC, ISplitInputLB>;

TEST(SearchTreeTest, ConstrainedEnumeration) {
  auto store = make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty)));
  populate_istore_n_vars(*store, 3, 0, 2);
  auto ipc = make_shared<IIPC, StandardAllocator>(IIPC(pty, store));
  x0_plus_x1_eq_x2(*ipc);
  auto split = make_shared<ISplitInputLB, StandardAllocator>(ISplitInputLB(split_ty, ipc, ipc));
  auto search_tree = IST(tty, ipc, split);
  EXPECT_FALSE2(search_tree.is_bot());
  EXPECT_FALSE2(search_tree.is_top());

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
  BInc has_changed = BInc::top();
  int iterations = 0;
  while(has_changed.guard()) {
    ++iterations;
    has_changed = BInc::bot();
    seq_refine(*ipc, has_changed);
    split->reset();
    seq_refine(*split, has_changed);
    if(search_tree.extract(sol)) {
      check_solution(sol, sols[solutions++]);
    }
    search_tree.refine(has_changed);
  }
  printf("iterations: %d\n", iterations);
  EXPECT_TRUE2(search_tree.is_top());
  EXPECT_FALSE2(search_tree.is_bot());
  has_changed = BInc::bot();
  seq_refine(*ipc, has_changed);
  search_tree.refine(has_changed);
  EXPECT_FALSE2(has_changed);
  EXPECT_TRUE2(search_tree.is_top());
  EXPECT_FALSE2(search_tree.is_bot());
  EXPECT_EQ(solutions, sols.size());
}
