// Copyright 2022 Pierre Talbot

#include "helper.hpp"

template <class A>
void fix_all_and_test(IStore& store, A& a) {
  for(int i = 2; i < 10; ++i) {
    LVar<StandardAllocator> x = "x ";
    x[1] = '0' + i;
    tell_store(store, make_v_op_z(x, EQ, 1), x, Itv(1, 1));
  }
  a.reset();
  seq_refine_check(a, BInc::bot());
  EXPECT_FALSE(a.project().has_value());
}

TEST(BranchTest, InputOrderTest) {
  auto store = make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty)));
  populate_10_vars(*store, 0, 10);
  InputOrder<IStore> input_order(store);
  input_order.interpret();
  EXPECT_FALSE(input_order.project().has_value());
  seq_refine_check(input_order);
  EXPECT_TRUE(input_order.project().has_value());
  EXPECT_EQ2(*input_order.project(), make_var(sty, 0));

  // Fix the second variable (in order) to "1". The first variable should still be selected.
  tell_store(*store, make_v_op_z(LVar<StandardAllocator>("x1"), EQ, 1), "x1", Itv(1, 1));
  input_order.reset();
  seq_refine_check(input_order);
  EXPECT_TRUE(input_order.project().has_value());
  EXPECT_EQ2(*input_order.project(), make_var(sty, 0));

  // Fix the first variable (in order) to "1". The third variable should now be selected.
  tell_store(*store, make_v_op_z(LVar<StandardAllocator>("x0"), EQ, 1), "x0", Itv(1, 1));
  input_order.reset();
  seq_refine_check(input_order);
  EXPECT_TRUE(input_order.project().has_value());
  EXPECT_EQ2(*input_order.project(), make_var(sty, 2));

  // Fix all the remaining variables.
  fix_all_and_test(*store, input_order);
}

template <class A>
void tell_store2(A& a, const typename A::TellType& tell) {
  BInc has_changed = BInc::bot();
  a.tell(tell, has_changed);
  EXPECT_EQ2(has_changed, BInc::top());
}

template <class A>
void split_and_test(IStore& store, A& a, AVar x, Itv expect_left, Itv expect_right) {
  auto branches = a.split(x);
  EXPECT_TRUE(branches.has_next());
  EXPECT_EQ(branches.size(), 2);
  EXPECT_FALSE(branches.is_pruned());
  auto snapshot = store.snapshot();
  EXPECT_EQ2(store.project(x), meet(expect_left, expect_right));
  tell_store2(store, branches.next());
  EXPECT_EQ2(store.project(x), expect_left);
  EXPECT_TRUE(branches.has_next());
  EXPECT_FALSE(branches.is_pruned());
  EXPECT_EQ(branches.size(), 2);
  store.restore(snapshot);
  tell_store2(store, branches.next());
  EXPECT_EQ2(store.project(x), expect_right);
  EXPECT_FALSE(branches.has_next());
  EXPECT_FALSE(branches.is_pruned());
  EXPECT_EQ(branches.size(), 2);
  branches.prune();
  EXPECT_FALSE(branches.has_next());
  EXPECT_TRUE(branches.is_pruned());
  store.restore(snapshot);
  EXPECT_EQ2(store.project(x), meet(expect_left, expect_right));
}

TEST(BranchTest, LowerBoundTest) {
  auto store = make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty)));
  populate_10_vars(*store, 0, 10);
  LowerBound<IStore> lb(store);
  split_and_test(*store, lb, make_var(sty, 0), Itv(0,0), Itv(1,10));
}
