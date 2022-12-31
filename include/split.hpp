// Copyright 2022 Pierre Talbot

#ifndef SPLIT_HPP
#define SPLIT_HPP

#include "branch.hpp"
#include "logic/logic.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

namespace lala {

template <class A, class VariableOrder, class ValueOrder>
class Split {
public:
  using allocator_type = typename A::allocator_type;
  using branch_type = typename ValueOrder::branch_type;

private:
  AType atype;
  VariableOrder var_order;
  ValueOrder val_order;

public:
  CUDA Split(AType atype, VariableOrder&& var_order, ValueOrder&& val_order)
  : atype(atype), var_order(std::move(var_order)), val_order(std::move(val_order)) {}

  template<class A2, class VarO2, class ValO2>
  CUDA Split(const Split<A2, VarO2, ValO2>& other, AbstractDeps<allocator_type>& deps)
   : atype(other.atype), var_order(other.var_order, deps), val_order(other.val_order, deps) {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA local::BInc is_top() const {
    return var_order.is_top();
  }

  /** Later on, we might pass to interpret a splitting strategy. */
  template <class Env>
  CUDA void interpret_in(Env& env) {
    var_order.interpret_in(env);
  }

  CUDA int num_refinements() const {
    return var_order.num_refinements();
  }

  template <class Mem>
  CUDA void refine(int i, BInc<Mem>& has_changed) {
    var_order.refine(i, has_changed);
  }

  CUDA void reset() {
    var_order.reset();
  }

  CUDA thrust::optional<AVar> project() const {
    return var_order.project();
  }

  template <class Env>
  CUDA branch_type split(Env& env) {
    auto x = project();
    if(x.has_value()) {
      return val_order.split(*x, env);
    }
    return branch_type{};
  }
};

}

#endif