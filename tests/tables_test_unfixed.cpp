// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/tables.hpp"
#include "lala/fixpoint.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using IStore = VStore<Itv, standard_allocator>;
using ITables = Tables<IStore>;

using ZF = local::ZFlat;
using IStore = VStore<Itv, standard_allocator>;
using FTables = Tables<IStore, ZF>;

template <class L>
void test_extract(const L& tables, bool is_ua) {
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  L copy1(tables, deps);
  EXPECT_EQ(tables.is_extractable(), is_ua);
  if(tables.is_extractable()) {
    tables.extract(copy1);
    EXPECT_EQ(tables.is_top(), copy1.is_top());
    EXPECT_EQ(tables.is_bot(), copy1.is_bot());
    for(int i = 0; i < tables.vars(); ++i) {
      EXPECT_EQ(tables[i], copy1[i]);
    }
  }
}

template<class L>
void refine_and_test(L& tables, int num_refine, const std::vector<Itv>& before, const std::vector<Itv>& after, bool is_ua, bool expect_changed = true) {
  EXPECT_EQ(tables.num_refinements(), num_refine);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(tables[i], before[i]) << "tables[" << i << "]";
  }
  local::B has_changed = false;
  GaussSeidelIteration{}.fixpoint(
    tables.num_refinements(),
    [&](size_t i) { return tables.refine(i); },
    has_changed);
  EXPECT_EQ(has_changed, expect_changed);
  for(int i = 0; i < after.size(); ++i) {
    EXPECT_EQ(tables[i], after[i]) << "tables[" << i << "]";
  }
  test_extract(tables, is_ua);
}

template<class L>
void refine_and_test(L& tables, int num_refine, const std::vector<Itv>& before_after, bool is_ua = false) {
  refine_and_test(tables, num_refine, before_after, before_after, is_ua, false);
}

/**
 *     x
 *  [1..1]
 *  [2..2]
 *  [3..3]
*/
TEST(ITablesTest, SingleConstantTable1) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 1..3: x;\
    constraint bool_or(bool_or(\
      int_eq(x, 1), int_eq(x, 2)), int_eq(x, 3), true);");
  refine_and_test(tables, 1 + 3*1, {Itv(1,3)});
  tables.subdomain()->tell(0, Itv(1,2));
  // tables changes internally but no domain could be pruned.
  refine_and_test(tables, 1 + 3*1, {Itv(1,2)}, {Itv(1,2)}, false);
  tables.subdomain()->tell(0, Itv(1,1));
  refine_and_test(tables, 1 + 3*1, {Itv(1,1)}, {Itv(1,1)}, true);
}

/**
 *     x      y     z
 *  [1..1] [1..1] [1..1]
 *  [2..2] [2..2] [2..2]
 *  [3..3] [3..3] [3..3]
*/
TEST(ITablesTest, SingleConstantTable2) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  tables.subdomain()->tell(1, Itv(1,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,2), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
  tables.subdomain()->tell(2, Itv(2,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,2), Itv(1,2), Itv(2,2)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

TEST(ITablesTest, SingleConstantTable2MeetOp) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 0..10: x; var 1..4: y; var 0..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(tables, 3 + 3*3, {Itv(0,10), Itv(1,4), Itv(0,3)}, {Itv(1,3), Itv(1,3), Itv(1,3)}, false);
}

TEST(ITablesTest, SingleConstantTable2AskOp1) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 1..2: x; var 1..3: y; var 2..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(tables, 3 + 3*3, {Itv(1,2), Itv(1,3), Itv(2,3)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

TEST(ITablesTest, SingleConstantTable2AskOp2) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 1..2: x; var 1..3: y; var 1..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(tables, 3 + 3*3, {Itv(1,2), Itv(1,3), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
}

/** Just to try with the nary version of bool_and and bool_or. */
TEST(ITablesTest, SingleConstantTable2b) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint nbool_or(\
      nbool_and(int_eq(x, 1), int_eq(y, 1), int_eq(z, 1)),\
      nbool_and(int_eq(x, 2), int_eq(y, 2), int_eq(z, 2)),\
      nbool_and(int_eq(x, 3), int_eq(y, 3), int_eq(z, 3)));");
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  tables.subdomain()->tell(1, Itv(1,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,2), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
  tables.subdomain()->tell(2, Itv(2,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,2), Itv(1,2), Itv(2,2)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

/**
 *     x      y     z
 *     1      1     1
 *     2      2     2
 *     3      3     3
*/
TEST(FTablesTest, SingleFlatTable1) {
  FTables tables = create_and_interpret_and_tell<FTables>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint nbool_or(\
      nbool_and(int_eq(x, 1), int_eq(y, 1), int_eq(z, 1)),\
      nbool_and(int_eq(x, 2), int_eq(y, 2), int_eq(z, 2)),\
      nbool_and(int_eq(x, 3), int_eq(y, 3), int_eq(z, 3)));");
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  tables.subdomain()->tell(1, Itv(1,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,2), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
  tables.subdomain()->tell(2, Itv(2,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,2), Itv(1,2), Itv(2,2)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

/**
 *     x      y     z
 *     *      1     *
 *     2      2     2
 *     *      3     *
*/
TEST(FTablesTest, SingleShortFlatTable1) {
  FTables tables = create_and_interpret_and_tell<FTables>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint nbool_or(\
      int_eq(y, 1),\
      nbool_and(int_eq(x, 2), int_eq(y, 2), int_eq(z, 2)),\
      int_eq(y, 3));");
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  tables.subdomain()->tell(1, Itv(2,3));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(2,3), Itv(1,3)}, {Itv(1,3), Itv(2,3), Itv(1,3)}, false);
  auto snap = tables.snapshot();
  tables.subdomain()->tell(2, Itv(2,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(2,3), Itv(2,2)});
  tables.subdomain()->tell(1, Itv(3,3));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(3,3), Itv(2,2)}, {Itv(1,3), Itv(3,3), Itv(2,2)}, true);
  tables.restore(snap);
  tables.subdomain()->tell(1, Itv(2,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(2,2), Itv(1,3)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

/**
 *     *   [1..1] [1..1]
 *  [2..2] [2..2] [2..2]
 *  [3..3] [3..3]   *
*/
TEST(ITablesTest, SingleShortTable1) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(y, 1), int_eq(z, 1)),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), int_eq(y, 3)), true);");
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  tables.subdomain()->tell(2, Itv(1,2));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,3), Itv(1,2)});
  tables.subdomain()->tell(0, Itv(2,3));
  refine_and_test(tables, 3 + 3*3, {Itv(2,3), Itv(1,3), Itv(1,2)});
  tables.subdomain()->tell(1, Itv(1,1));
  refine_and_test(tables, 3 + 3*3, {Itv(2,3), Itv(1,1), Itv(1,2)}, {Itv(2,3), Itv(1,1), Itv(1,1)}, true);
}

/**
 *     x      y     z
 *  [0..3] [1..3] [0..2]
 *  [2..4] [1..4] [2..2]
 *  [5..7] [1..9] [3..3]
*/
TEST(ITablesTest, SingleSmartTable1) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 0..8: x; var 0..8: y; var 0..8: z;\
    constraint nbool_or(\
      nbool_and(int_ge(x, 0), int_le(x, 3), int_ge(y, 1), int_le(y, 3), int_ge(z, 0), int_le(z, 2)),\
      nbool_and(int_ge(x, 2), int_le(x, 4), int_ge(y, 1), int_le(y, 4), int_eq(z, 2)),\
      nbool_and(int_ge(x, 5), int_le(x, 7), int_ge(y, 1), int_le(y, 9), int_eq(z, 3)));");
  refine_and_test(tables, 3 + 3*3, {Itv(0,8), Itv(0,8), Itv(0,8)}, {Itv(0,7), Itv(1,8), Itv(0,3)}, false);
  tables.subdomain()->tell(0, Itv(1,3));
  refine_and_test(tables, 3 + 3*3, {Itv(1,3), Itv(1,8), Itv(0,3)}, {Itv(1,3), Itv(1,4), Itv(0,2)}, false);
  tables.subdomain()->tell(0, Itv(1,1));
  refine_and_test(tables, 3 + 3*3, {Itv(1,1), Itv(1,4), Itv(0,2)}, {Itv(1,1), Itv(1,3), Itv(0,2)}, true);
}

/**
 *     x      y     z       |     y      z      w
 *  [0..5] [0..4] [1..6]       [6..6] [8..8] [5..9]
 *  [1..6] [0..5] [2..7]       [0..0] [1..1] [0..5]
 *  [2..7] [1..6] [3..8]
*/
TEST(ITablesTest, MultiSmartTables1) {
  ITables tables = create_and_interpret_and_tell<ITables>(
    "var 0..9: x; var 0..9: y; var 0..9: z; var 0..9: w;\
    constraint nbool_or(\
      nbool_and(int_ge(x, 0), int_le(x, 5), int_ge(y, 0), int_le(y, 4), int_ge(z, 1), int_le(z, 6)),\
      nbool_and(int_ge(x, 1), int_le(x, 6), int_ge(y, 0), int_le(y, 5), int_ge(z, 2), int_le(z, 7)),\
      nbool_and(int_ge(x, 2), int_le(x, 7), int_ge(y, 1), int_le(y, 6), int_ge(z, 3), int_le(z, 8)));\
    constraint nbool_or(\
      nbool_and(int_ge(y, 6), int_le(y, 6), int_ge(z, 8), int_le(z, 8), int_ge(w, 5), int_le(w, 9)),\
      nbool_and(int_ge(y, 0), int_le(y, 0), int_ge(z, 1), int_le(z, 1), int_ge(w, 0), int_le(w, 5)));");

  refine_and_test(tables, 3 + 3*3 + 3 + 2*3, {Itv(0,9), Itv(0,9), Itv(0,9), Itv(0,9)}, {Itv(0,7), Itv(0,6), Itv(1,8), Itv(0,9)}, false);
  tables.subdomain()->tell(3, Itv(5,9));
  refine_and_test(tables, 3 + 3*3 + 3 + 2*3, {Itv(0,7), Itv(0,6), Itv(1,8), Itv(5,9)});
  tables.subdomain()->tell(3, Itv(6,9));
  refine_and_test(tables, 3 + 3*3 + 3 + 2*3, {Itv(0,7), Itv(0,6), Itv(1,8), Itv(6,9)}, {Itv(2,7), Itv(6,6), Itv(8,8), Itv(6,9)}, true);
}
