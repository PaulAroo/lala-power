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
    EXPECT_EQ2(a.project(make_var(sty, i)), solution[i]);
  }
}

using SF = SFormula<StandardAllocator>;

void test_unconstrained_bab(SF::Mode mode) {
  auto store = make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty)));
  populate_istore_n_vars(*store, 3, 0, 2);
  auto split = make_shared<SplitInputLB, StandardAllocator>(SplitInputLB(split_ty, store, store));
  auto search_tree = make_shared<ST, StandardAllocator>(tty, store, split);
  auto bab = BAB_(bab_ty, search_tree);

  // Find solution optimizing x1.
  auto min_x1 = bab.interpret(SF(F::make_true(), mode, "x1"));
  EXPECT_TRUE(min_x1.has_value());
  BInc has_changed = BInc::bot();
  bab.tell(std::move(*min_x1), has_changed);
  EXPECT_TRUE2(has_changed);

  int iterations = 0;
  while(!bab.extract(bab) && has_changed.guard()) {
    iterations++;
    has_changed = BInc::bot();
    split->reset();
    // Compute \f$ pop \circ push \circ split \circ bab \f$.
    bab.refine(has_changed);
    seq_refine(*split, has_changed);
    search_tree->refine(has_changed);
  }
  // Find the optimum in the root node since they are no constraint...
  check_solution(bab.optimum(), {Itv(0,2),Itv(0,2),Itv(0,2)});
  // With a input-order smallest first strat, the fixed point is reached after 1 iteration.
  EXPECT_EQ(iterations, 1);

  EXPECT_TRUE2(search_tree->is_top());

  // One more iteration to check idempotency.
  has_changed = BInc::bot();
  split->reset();
  bab.refine(has_changed);
  seq_refine(*split, has_changed);
  search_tree->refine(has_changed);
  EXPECT_FALSE2(has_changed);
}

TEST(BABTest, UnconstrainedOptimization) {
  test_unconstrained_bab(SF::Mode::MINIMIZE);
  test_unconstrained_bab(SF::Mode::MAXIMIZE);
}
