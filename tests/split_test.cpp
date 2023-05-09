// Copyright 2022 Pierre Talbot

#include "helper.hpp"

template <class A, class Env>
void fix_all_and_test(IStore& store, A& a, Env& env) {
  for(int i = 2; i < store.vars(); ++i) {
    LVar<StandardAllocator> x = "x ";
    x[1] = '0' + i + 1;
    interpret_and_tell(store, ("constraint int_eq(" + x + ", 1);").data(), env);
  }
  a.reset();
  seq_refine_check(a, local::BInc::bot());
  EXPECT_FALSE(a.project().has_value());
}

TEST(BranchTest, InputOrderTest) {
  VarEnv<StandardAllocator> env;
  shared_ptr<IStore, StandardAllocator> store =
    make_shared<IStore, StandardAllocator>(std::move(
      interpret_tell_to<IStore>("var int: x1; var int: x2; var int: x3; var int: x4; var int: x5;\
        constraint int_ge(x1, 0); constraint int_le(x1, 10);\
        constraint int_ge(x2, 0); constraint int_le(x2, 10);\
        constraint int_ge(x3, 0); constraint int_le(x3, 10);\
        constraint int_ge(x4, 0); constraint int_le(x4, 10);\
        constraint int_ge(x5, 0); constraint int_le(x5, 10);", env)));

  InputOrder<IStore> input_order(store);
  input_order.interpret_tell_in(env);
  EXPECT_FALSE(input_order.project().has_value());
  seq_refine_check(input_order);
  EXPECT_TRUE(input_order.project().has_value());
  EXPECT_EQ(*input_order.project(), AVar(sty, 0));

  // Fix the second variable (in order) to "1". The first variable should still be selected.
  interpret_and_tell(*store, "constraint int_eq(x2, 1);", env);
  input_order.reset();
  seq_refine_check(input_order);
  EXPECT_TRUE(input_order.project().has_value());
  EXPECT_EQ(*input_order.project(), AVar(sty, 0));

  // Fix the first variable (in order) to "1". The third variable should now be selected.
  interpret_and_tell(*store, "constraint int_eq(x1, 1);", env);
  input_order.reset();
  seq_refine_check(input_order);
  EXPECT_TRUE(input_order.project().has_value());
  EXPECT_EQ(*input_order.project(), AVar(sty, 2));

  // Fix all the remaining variables.
  fix_all_and_test(*store, input_order, env);
}

template <class A, class TellType>
void tell_store2(A& a, const TellType& tell) {
  local::BInc has_changed;
  a.tell(tell, has_changed);
  EXPECT_EQ(has_changed, local::BInc::top());
}

template <class A, class Env>
void split_and_test(IStore& store, A& a, Env& env, AVar x, Itv expect_left, Itv expect_right) {
  auto branches = a.split(x, env);
  EXPECT_TRUE(branches.has_next());
  EXPECT_EQ(branches.size(), 2);
  EXPECT_FALSE(branches.is_pruned());
  auto snapshot = store.snapshot();
  EXPECT_EQ(store.project(x), meet(expect_left, expect_right));
  tell_store2(store, branches.next());
  EXPECT_EQ(store.project(x), expect_left);
  EXPECT_TRUE(branches.has_next());
  EXPECT_FALSE(branches.is_pruned());
  EXPECT_EQ(branches.size(), 2);
  store.restore(snapshot);
  tell_store2(store, branches.next());
  EXPECT_EQ(store.project(x), expect_right);
  EXPECT_FALSE(branches.has_next());
  EXPECT_FALSE(branches.is_pruned());
  EXPECT_EQ(branches.size(), 2);
  branches.prune();
  EXPECT_FALSE(branches.has_next());
  EXPECT_TRUE(branches.is_pruned());
  store.restore(snapshot);
  EXPECT_EQ(store.project(x), meet(expect_left, expect_right));
}

TEST(BranchTest, LowerBoundTest) {
  VarEnv<StandardAllocator> env;
  shared_ptr<IStore, StandardAllocator> store =
    make_shared<IStore, StandardAllocator>(std::move(
      interpret_tell_to<IStore>("var int: x1; var int: x2; var int: x3; var int: x4; var int: x5;\
        constraint int_ge(x1, 0); constraint int_le(x1, 10);\
        constraint int_ge(x2, 0); constraint int_le(x2, 10);\
        constraint int_ge(x3, 0); constraint int_le(x3, 10);\
        constraint int_ge(x4, 0); constraint int_le(x4, 10);\
        constraint int_ge(x5, 0); constraint int_le(x5, 10);", env)));
  LowerBound<IStore> lb(store);
  split_and_test(*store, lb, env, AVar(sty, 0), Itv(0,0), Itv(1,10));
}
