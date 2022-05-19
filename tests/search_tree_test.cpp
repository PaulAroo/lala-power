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
  auto split = make_shared<SplitInputLB, StandardAllocator>(SplitInputLB(store, store));
  auto search_tree = ST(tty, store, split);
  EXPECT_TRUE2(search_tree.is_bot());
  EXPECT_FALSE2(search_tree.is_top());

  int solutions = 0;
  for(int x0 = 0; x0 < 3; ++x0) {
    for(int x1 = 0; x1 < 3; ++x1) {
      for(int x2 = 0; x2 < 3; ++x2) {
        // Going down a branch of the seach tree.
        bool leaf = false;
        while(!leaf) {
          BInc has_changed = BInc::bot();
          split->reset();
          seq_refine(*split, has_changed);
          // All variables are supposed to be assigned.
          if(!has_changed.value()) {
            check_solution(*store, {x0, x1, x2});
            solutions++;
            leaf = true;
          }
          search_tree.refine(has_changed);
          if(x0 == 2 && x1 == 2 && x2 == 2 && leaf) {
            EXPECT_FALSE2(has_changed);
          }
          else {
            EXPECT_TRUE2(has_changed);
          }
        }
      }
    }
  }
  EXPECT_TRUE2(search_tree.is_top());
  EXPECT_FALSE2(search_tree.is_bot());
  EXPECT_EQ(solutions, 3*3*3);
}
