// Copyright 2022 Pierre Talbot

#ifndef SPLIT_HPP
#define SPLIT_HPP

#include "ast.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"
#include "branch.hpp"

namespace lala {

template <class A, class VariableOrder, class ValueOrder>
class Split {
public:
  using Allocator = typename A::Allocator;
  using BranchType = typename ValueOrder::BranchType;

private:
  AType uid_;
  VariableOrder var_order;
  ValueOrder val_order;

public:
  CUDA Split(AType uid, VariableOrder&& var_order, ValueOrder&& val_order)
  : uid_(uid), var_order(std::move(var_order)), val_order(std::move(val_order)) {}

  template<class A2, class VarO2, class ValO2>
  CUDA Split(const Split<A2, VarO2, ValO2>& other, AbstractDeps<Allocator>& deps)
   : uid_(other.uid_), var_order(other.var_order, deps), val_order(other.val_order, deps) {}

  CUDA AType uid() const {
    return uid_;
  }

  CUDA BInc is_top() const {
    return var_order.is_top();
  }

  /** Later on, we might pass to interpret a splitting strategy. */
  CUDA void interpret() {
    var_order.interpret();
  }

  CUDA int num_refinements() const {
    return var_order.num_refinements();
  }

  CUDA void refine(int i, BInc& has_changed) {
    var_order.refine(i, has_changed);
  }

  CUDA void reset() {
    var_order.reset();
  }

  CUDA thrust::optional<AVar> project() const {
    return var_order.project();
  }

  CUDA BranchType split() {
    auto x = project();
    if(x.has_value()) {
      return val_order.split(*x);
    }
    return BranchType{};
  }
};

}

#endif