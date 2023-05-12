// Copyright 2022 Pierre Talbot

#include "helper.hpp"
#include "battery/memory.hpp"

void apply_branch_and_test(
  shared_ptr<IStore, standard_allocator> store,
  const IStore::tell_type<standard_allocator>& branch,
  int var_idx,
  Itv expected)
{
  auto snapshot = store->snapshot();
  store->tell(branch);
  EXPECT_EQ((*store)[var_idx], expected);
  store->restore(snapshot);
}

void test_strategy(
  const std::string& variable_order,
  const std::string& value_order,
  int var_idx,
  Itv left,
  Itv right)
{
  FlatZincOutput<standard_allocator> output;
  lala::impl::FlatZincParser<standard_allocator> parser(output);
  auto f = parser.parse("var 1..1: x1; var 3..8: x2; var 5..5: x3; var 4..6: x4; var 0..7: x5; var 2..10: x6; var 2..2: x7;");
  EXPECT_TRUE(f);
  VarEnv<standard_allocator> env;
  auto store_res = IStore::interpret_tell(*f, env);
  shared_ptr<IStore, standard_allocator> store =
    make_shared<IStore, standard_allocator>(store_res.value());
  shared_ptr<SplitStrategy<IStore>> split =
    make_shared<SplitStrategy<IStore>, standard_allocator>(env.extends_abstract_dom(), store);
  auto strat = parser.parse(
    "solve::int_search([x1,x2,x3,x4,x5,x6,x7], " + variable_order + ", " + value_order + ", complete) satisfy;");
  EXPECT_TRUE(strat);
  auto split_res = split->interpret_tell_in(*strat, env);
  EXPECT_TRUE(split_res.has_value());
  split->tell(split_res.value());
  auto branches = split->split();
  auto left_branch = branches.next();
  EXPECT_EQ(branches.size(), 2);
  EXPECT_EQ(left_branch.size(), 1);
  EXPECT_EQ(left_branch[0].idx, var_idx);
  apply_branch_and_test(store, left_branch, var_idx, left);
  auto right_branch = branches.next();
  EXPECT_EQ(branches.size(), 2);
  EXPECT_EQ(right_branch.size(), 1);
  EXPECT_EQ(right_branch[0].idx, var_idx);
  apply_branch_and_test(store, right_branch, var_idx, right);
}

TEST(BranchTest, InputOrderTest) {
  // Note that for intervals, we cannot exclude values in the middle... so this indomain_median creates constraints that are uninterpretable.

  test_strategy("input_order", "indomain_min", 1, Itv(3, 3), Itv(4, 8));
  test_strategy("input_order", "indomain_max", 1, Itv(8, 8), Itv(3, 7));
  test_strategy("input_order", "indomain_split", 1, Itv(3, 5), Itv(6, 8));
  test_strategy("input_order", "indomain_reverse_split", 1, Itv(6, 8), Itv(3, 5));

  test_strategy("first_fail", "indomain_min", 3, Itv(4, 4), Itv(5, 6));
  test_strategy("first_fail", "indomain_max", 3, Itv(6, 6), Itv(4, 5));
  test_strategy("first_fail", "indomain_split", 3, Itv(4, 5), Itv(6, 6));
  test_strategy("first_fail", "indomain_reverse_split", 3, Itv(6, 6), Itv(4, 5));

  test_strategy("anti_first_fail", "indomain_min", 5, Itv(2, 2), Itv(3, 10));
  test_strategy("anti_first_fail", "indomain_max", 5, Itv(10, 10), Itv(2, 9));
  test_strategy("anti_first_fail", "indomain_split", 5, Itv(2, 6), Itv(7, 10));
  test_strategy("anti_first_fail", "indomain_reverse_split", 5, Itv(7, 10), Itv(2, 6));

  test_strategy("smallest", "indomain_min", 4, Itv(0, 0), Itv(1, 7));
  test_strategy("smallest", "indomain_max", 4, Itv(7, 7), Itv(0, 6));
  test_strategy("smallest", "indomain_split", 4, Itv(0, 3), Itv(4, 7));
  test_strategy("smallest", "indomain_reverse_split", 4, Itv(4, 7), Itv(0, 3));

  test_strategy("largest", "indomain_min", 5, Itv(2, 2), Itv(3, 10));
  test_strategy("largest", "indomain_max", 5, Itv(10, 10), Itv(2, 9));
  test_strategy("largest", "indomain_split", 5, Itv(2, 6), Itv(7, 10));
  test_strategy("largest", "indomain_reverse_split", 5, Itv(7, 10), Itv(2, 6));
}

using AItv = Interval<ZInc<int, battery::atomic_memory<standard_allocator>>>;
using AIStore = VStore<AItv, standard_allocator>;

TEST(BranchTest, CopySplitStrategy) {
  VarEnv<standard_allocator> env;
  shared_ptr<IStore, standard_allocator> store =
    make_shared<IStore, standard_allocator>(env.extends_abstract_dom(), 0);
  shared_ptr<SplitStrategy<IStore>> split =
    make_shared<SplitStrategy<IStore>, standard_allocator>(env.extends_abstract_dom(), store);
  AbstractDeps<standard_allocator> deps;
  auto r = deps.template clone<SplitStrategy<AIStore>>(split);
}
