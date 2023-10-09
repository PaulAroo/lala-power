// Copyright 2022 Pierre Talbot

#include "lala/search_tree.hpp"
#include "helper.hpp"

using ST = SearchTree<IStore, SplitStrategy<IStore>>;

template <class A>
void check_solution(A& a, vector<int> solution) {
  for(int i = 0; i < solution.size(); ++i) {
    EXPECT_EQ(a.project(AVar(sty, i)), Itv(solution[i]));
  }
}

bool all_assigned(const IStore& a) {
  for(int i = 0; i < a.vars(); ++i) {
    if(a[i].lb() != a[i].ub()) {
      return false;
    }
  }
  return true;
}

TEST(SearchTreeTest, EnumerationSolution) {
  FlatZincOutput<standard_allocator> output(standard_allocator{});
  lala::impl::FlatZincParser<standard_allocator> parser(output);
  auto f = parser.parse("array[1..3] of var 0..2: a;\
    solve::int_search(a, input_order, indomain_min, complete) satisfy;");
  EXPECT_TRUE(f);
  VarEnv<standard_allocator> env;
  auto store = make_shared<IStore, standard_allocator>(env.extends_abstract_dom(), 3);
  auto split = make_shared<SplitStrategy<IStore>, standard_allocator>(env.extends_abstract_dom(), store);
  auto search_tree = ST(env.extends_abstract_dom(), store, split);

  EXPECT_TRUE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  auto st_res = search_tree.interpret_tell_in(*f, env);
  EXPECT_TRUE(st_res.has_value());
  local::BInc has_changed;
  search_tree.tell(st_res.value(), has_changed);
  EXPECT_TRUE(has_changed);

  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  AbstractDeps<standard_allocator> deps{standard_allocator{}};
  ST sol(search_tree, deps);

  int solutions = 0;
  for(int x1 = 0; x1 < 3; ++x1) {
    for(int x2 = 0; x2 < 3; ++x2) {
      for(int x3 = 0; x3 < 3; ++x3) {
        // Going down a branch of the search tree until all variables are assigned.
        do {
          has_changed.dtell_bot();
          search_tree.refine(has_changed);
          EXPECT_TRUE(has_changed);
        } while(!all_assigned(*store));
        // There is no constraint so we are always navigating the under-approximated space.
        EXPECT_TRUE(search_tree.is_extractable());
        search_tree.extract(sol);
        // All variables are supposed to be assigned if nothing changed.
        check_solution(sol, {x1, x2, x3});
        solutions++;
      }
    }
  }
  EXPECT_FALSE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  has_changed.dtell_bot();
  search_tree.refine(has_changed);
  EXPECT_TRUE(has_changed);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  has_changed.dtell_bot();
  search_tree.refine(has_changed);
  EXPECT_FALSE(has_changed);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_EQ(solutions, 3*3*3);
}

using IST = SearchTree<IPC, SplitStrategy<IPC>>;

TEST(SearchTreeTest, ConstrainedEnumeration) {
  FlatZincOutput<standard_allocator> output(standard_allocator{});
  lala::impl::FlatZincParser<standard_allocator> parser(output);
  auto f = parser.parse("array[1..3] of var 0..2: a;\
    constraint int_plus(a[1], a[2], a[3]);\
    solve::int_search(a, input_order, indomain_min, complete) satisfy;");
  EXPECT_TRUE(f);
  VarEnv<standard_allocator> env;
  auto store = make_shared<IStore, standard_allocator>(env.extends_abstract_dom(), 3);
  auto ipc = make_shared<IPC, standard_allocator>(IPC(env.extends_abstract_dom(), store));
  auto split = make_shared<SplitStrategy<IPC>, standard_allocator>(env.extends_abstract_dom(), ipc);
  auto search_tree = IST(env.extends_abstract_dom(), ipc, split);

  EXPECT_TRUE(search_tree.is_bot());
  EXPECT_FALSE(search_tree.is_top());

  auto st_res = search_tree.interpret_tell_in(*f, env);
  EXPECT_TRUE(st_res.has_value());
  local::BInc has_changed;
  search_tree.tell(st_res.value(), has_changed);
  EXPECT_TRUE(has_changed);

  AbstractDeps<standard_allocator> deps{standard_allocator{}};
  IST sol(search_tree, deps);

  int solutions = 0;
  vector<vector<int>> sols = {
    {0, 0, 0},
    {0, 1, 1},
    {0, 2, 2},
    {1, 0, 1},
    {1, 1, 2},
    {2, 0, 2}} ;
  has_changed = true;
  int iterations = 0;
  while(has_changed) {
    ++iterations;
    has_changed = false;
    GaussSeidelIteration{}.fixpoint(*ipc, has_changed);
    if(all_assigned(*store) && search_tree.is_extractable()) {
      search_tree.extract(sol);
      check_solution(sol, sols[solutions++]);
    }
    search_tree.refine(has_changed);
    // std::cout << *store << std::endl;
  }
  EXPECT_EQ(iterations, 12);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  has_changed = local::BInc::bot();
  GaussSeidelIteration{}.fixpoint(*ipc, has_changed);
  search_tree.refine(has_changed);
  EXPECT_FALSE(has_changed);
  EXPECT_TRUE(search_tree.is_top());
  EXPECT_FALSE(search_tree.is_bot());
  EXPECT_EQ(solutions, sols.size());
}
