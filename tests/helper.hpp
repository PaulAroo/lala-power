// Copyright 2022 Pierre Talbot

#ifndef HELPER_HPP
#define HELPER_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "abstract_testing.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/pc.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"

#include "lala/value_order.hpp"
#include "lala/variable_order.hpp"
#include "lala/split.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<StandardAllocator>;

static LVar<StandardAllocator> var_x0 = "x0";
static LVar<StandardAllocator> var_x1 = "x1";
static LVar<StandardAllocator> var_x2 = "x2";
static LVar<StandardAllocator> var_z = "z";
static LVar<StandardAllocator> var_b = "b";

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using IStore = VStore<Itv, StandardAllocator>;
using IPC = PC<IStore>; // Interval Propagators Completion

const AType sty = 0;
const AType pty = 1;
const AType tty = 2;
const AType split_ty = 3;
const AType bab_ty = 4;

template <class A>
void seq_refine_check(A& a, local::BInc expect_changed = true) {
  local::BInc has_changed;
  GaussSeidelIteration{}.iterate(a, has_changed);
  EXPECT_EQ(has_changed, expect_changed);
}

#endif