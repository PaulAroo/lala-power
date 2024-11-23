// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/table.hpp"
#include "lala/fixpoint.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using IStore = VStore<Itv, standard_allocator>;
using ITable = Table<IStore>;

using ZF = local::ZFlat;
using IStore = VStore<Itv, standard_allocator>;
using FTable = Table<IStore, ZF>;

template <class L>
void test_extract(const L& table, bool is_ua) {
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  L copy1(table, deps);
  EXPECT_EQ(table.is_extractable(), is_ua);
  if(table.is_extractable()) {
    table.extract(copy1);
    EXPECT_EQ(table.is_top(), copy1.is_top());
    EXPECT_EQ(table.is_bot(), copy1.is_bot());
    for(int i = 0; i < table.vars(); ++i) {
      EXPECT_EQ(table[i], copy1[i]);
    }
  }
}

template<class L>
void refine_and_test(L& table, int num_refine, const std::vector<Itv>& before, const std::vector<Itv>& after, bool is_ua, bool expect_changed = true) {
  EXPECT_EQ(table.num_refinements(), num_refine);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(table[i], before[i]) << "table[" << i << "]";
  }
  local::B has_changed = false;
  GaussSeidelIteration{}.fixpoint(
    table.num_refinements(),
    [&](size_t i) { return table.refine(i); },
    has_changed
  );
  EXPECT_EQ(has_changed, expect_changed);
  for(int i = 0; i < after.size(); ++i) {
    EXPECT_EQ(table[i], after[i]) << "table[" << i << "]";
  }
  test_extract(table, is_ua);
}

template<class L>
void refine_and_test(L& table, int num_refine, const std::vector<Itv>& before_after, bool is_ua = false) {
  refine_and_test(table, num_refine, before_after, before_after, is_ua, false);
}

/**
 *     x      y     z
 *  [1..1] [1..1] [1..1]
 *  [2..2] [2..2] [2..2]
 *  [3..3] [3..3] [3..3]
*/
TEST(ITableTest, SingleConstantTable2) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(table, 3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  table.subdomain()->tell(1, Itv(1,2));
  refine_and_test(table, 3, {Itv(1,3), Itv(1,2), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
  table.subdomain()->tell(2, Itv(2,2));
  refine_and_test(table, 3, {Itv(1,2), Itv(1,2), Itv(2,2)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

TEST(ITableTest, SingleConstantTable2MeetOp) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 0..10: x; var 1..4: y; var 0..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(table, 3, {Itv(0,10), Itv(1,4), Itv(0,3)}, {Itv(1,3), Itv(1,3), Itv(1,3)}, false);
}

TEST(ITableTest, SingleConstantTable2AskOp1) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 1..2: x; var 1..3: y; var 2..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(table, 3, {Itv(1,2), Itv(1,3), Itv(2,3)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

TEST(ITableTest, SingleConstantTable2AskOp2) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 1..2: x; var 1..3: y; var 1..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(x, 1), bool_and(int_eq(y, 1), int_eq(z, 1))),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), bool_and(int_eq(y, 3), int_eq(z, 3))), true);");
  refine_and_test(table, 3, {Itv(1,2), Itv(1,3), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
}

/** Just to try with the nary version of bool_and and bool_or. */
TEST(ITableTest, SingleConstantTable2b) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint nbool_or(\
      nbool_and(int_eq(x, 1), int_eq(y, 1), int_eq(z, 1)),\
      nbool_and(int_eq(x, 2), int_eq(y, 2), int_eq(z, 2)),\
      nbool_and(int_eq(x, 3), int_eq(y, 3), int_eq(z, 3)));");
  refine_and_test(table, 3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  table.subdomain()->tell(1, Itv(1,2));
  refine_and_test(table, 3, {Itv(1,3), Itv(1,2), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
  table.subdomain()->tell(2, Itv(2,2));
  refine_and_test(table, 3, {Itv(1,2), Itv(1,2), Itv(2,2)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

/**
 *     x      y     z
 *     1      1     1
 *     2      2     2
 *     3      3     3
*/
TEST(FTableTest, SingleFlatTable1) {
  FTable table = create_and_interpret_and_tell<FTable>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint nbool_or(\
      nbool_and(int_eq(x, 1), int_eq(y, 1), int_eq(z, 1)),\
      nbool_and(int_eq(x, 2), int_eq(y, 2), int_eq(z, 2)),\
      nbool_and(int_eq(x, 3), int_eq(y, 3), int_eq(z, 3)));");
  refine_and_test(table, 3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  table.subdomain()->tell(1, Itv(1,2));
  refine_and_test(table, 3, {Itv(1,3), Itv(1,2), Itv(1,3)}, {Itv(1,2), Itv(1,2), Itv(1,2)}, false);
  table.subdomain()->tell(2, Itv(2,2));
  refine_and_test(table, 3, {Itv(1,2), Itv(1,2), Itv(2,2)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

/**
 *     x      y     z
 *     *      1     *
 *     2      2     2
 *     *      3     *
*/
TEST(FTableTest, SingleShortFlatTable1) {
  FTable table = create_and_interpret_and_tell<FTable>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint nbool_or(\
      int_eq(y, 1),\
      nbool_and(int_eq(x, 2), int_eq(y, 2), int_eq(z, 2)),\
      int_eq(y, 3));");
  refine_and_test(table, 3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  table.subdomain()->tell(1, Itv(2,3));
  refine_and_test(table, 3, {Itv(1,3), Itv(2,3), Itv(1,3)}, {Itv(1,3), Itv(2,3), Itv(1,3)}, false);
  auto snap = table.snapshot();
  table.subdomain()->tell(2, Itv(2,2));
  refine_and_test(table, 3, {Itv(1,3), Itv(2,3), Itv(2,2)});
  table.subdomain()->tell(1, Itv(3,3));
  refine_and_test(table, 3, {Itv(1,3), Itv(3,3), Itv(2,2)}, {Itv(1,3), Itv(3,3), Itv(2,2)}, true);
  table.restore(snap);
  table.subdomain()->tell(1, Itv(2,2));
  refine_and_test(table, 3, {Itv(1,3), Itv(2,2), Itv(1,3)}, {Itv(2,2), Itv(2,2), Itv(2,2)}, true);
}

/**
 *     *   [1..1] [1..1]
 *  [2..2] [2..2] [2..2]
 *  [3..3] [3..3]   *
*/
TEST(ITableTest, SingleShortTable1) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 1..3: x; var 1..3: y; var 1..3: z;\
    constraint bool_or(bool_or(\
      bool_and(int_eq(y, 1), int_eq(z, 1)),\
      bool_and(int_eq(x, 2), bool_and(int_eq(y, 2), int_eq(z, 2)))),\
      bool_and(int_eq(x, 3), int_eq(y, 3)), true);");
  refine_and_test(table, 3, {Itv(1,3), Itv(1,3), Itv(1,3)});
  table.subdomain()->tell(2, Itv(1,2));
  refine_and_test(table, 3, {Itv(1,3), Itv(1,3), Itv(1,2)});
  table.subdomain()->tell(0, Itv(2,3));
  refine_and_test(table, 3, {Itv(2,3), Itv(1,3), Itv(1,2)});
  table.subdomain()->tell(1, Itv(1,1));
  refine_and_test(table, 3, {Itv(2,3), Itv(1,1), Itv(1,2)}, {Itv(2,3), Itv(1,1), Itv(1,1)}, true);
}

/**
 *     x      y     z
 *  [0..3] [1..3] [0..2]
 *  [2..4] [1..4] [2..2]
 *  [5..7] [1..9] [3..3]
*/
TEST(ITableTest, SingleSmartTable1) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 0..8: x; var 0..8: y; var 0..8: z;\
    constraint nbool_or(\
      nbool_and(int_ge(x, 0), int_le(x, 3), int_ge(y, 1), int_le(y, 3), int_ge(z, 0), int_le(z, 2)),\
      nbool_and(int_ge(x, 2), int_le(x, 4), int_ge(y, 1), int_le(y, 4), int_eq(z, 2)),\
      nbool_and(int_ge(x, 5), int_le(x, 7), int_ge(y, 1), int_le(y, 9), int_eq(z, 3)));");
  refine_and_test(table, 3, {Itv(0,8), Itv(0,8), Itv(0,8)}, {Itv(0,7), Itv(1,8), Itv(0,3)}, false);
  table.subdomain()->tell(0, Itv(1,3));
  refine_and_test(table, 3, {Itv(1,3), Itv(1,8), Itv(0,3)}, {Itv(1,3), Itv(1,4), Itv(0,2)}, false);
  table.subdomain()->tell(0, Itv(1,1));
  refine_and_test(table, 3, {Itv(1,1), Itv(1,4), Itv(0,2)}, {Itv(1,1), Itv(1,3), Itv(0,2)}, true);
}

/**
 *     x      y    |     y      z
 *  [0..5] [0..4]     [0..5] [0..4]
 *  [1..6] [0..5]     [1..6] [0..5]
 *  [6..7] [6..6]     [6..7] [6..6]
*/
TEST(ITableTest, MultiSmartTables1) {
  ITable table = create_and_interpret_and_tell<ITable>(
    "var 0..9: x; var 0..9: y; var 0..9: z;\
    constraint nbool_or(\
      nbool_and(int_ge(x, 0), int_le(x, 5), int_ge(y, 0), int_le(y, 4)),\
      nbool_and(int_ge(x, 1), int_le(x, 6), int_ge(y, 0), int_le(y, 5)),\
      nbool_and(int_ge(x, 6), int_le(x, 7), int_ge(y, 6), int_le(y, 6)));\
    constraint nbool_or(\
      nbool_and(int_ge(y, 0), int_le(y, 5), int_ge(z, 0), int_le(z, 4)),\
      nbool_and(int_ge(y, 1), int_le(y, 6), int_ge(z, 0), int_le(z, 5)),\
      nbool_and(int_ge(y, 6), int_le(y, 7), int_ge(z, 6), int_le(z, 6)));");

  refine_and_test(table, 4, {Itv(0,9), Itv(0,9), Itv(0,9)}, {Itv(0,7), Itv(0,6), Itv(0,6)}, false);
  table.subdomain()->tell(0, Itv(0,1));
  refine_and_test(table, 4, {Itv(0,1), Itv(0,6), Itv(0,6)}, {Itv(0,1), Itv(0,5), Itv(0,5)}, false);
  table.subdomain()->tell(2, Itv(5,5));
  refine_and_test(table, 4, {Itv(0,1), Itv(0,5), Itv(5,5)}, {Itv(0,1), Itv(1,5), Itv(5,5)}, false);
  table.subdomain()->tell(0, Itv(0,0));
  refine_and_test(table, 4, {Itv(0,0), Itv(1,5), Itv(5,5)}, {Itv(0,0), Itv(1,4), Itv(5,5)}, true);
}
