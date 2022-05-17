// Copyright 2022 Pierre Talbot

#ifndef VARIABLE_ORDER_HPP
#define VARIABLE_ORDER_HPP

#include "ast.hpp"
#include "z.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

namespace lala {

template <class A>
class VariableOrder
{
public:
  using Allocator = typename A::Allocator;
  using TellType = typename A::TellType;
  using LVarArray = battery::vector<LVar<Allocator>, Allocator>;

private:
  battery::vector<AVar, Allocator> vars;
  battery::shared_ptr<A, Allocator> a;

public:
  CUDA VariableOrder(VariableOrder&&) = default;
  CUDA VariableOrder(battery::shared_ptr<A, Allocator> a) : a(a) {
    const auto& env = a->environment();
    vars.reserve(env.size());
    for(int i = 0; i < env.size(); ++i) {
      vars.push_back(make_var(a->ad_uid(), i));
    }
  }

  CUDA VariableOrder(battery::shared_ptr<A, Allocator> a, const LVarArray& lvars) : a(a) {
    const auto& env = a->environment();
    vars.reserve(lvars.size());
    for(int i = 0; i < lvars.size(); ++i) {
      vars.push_back(*(env.to_avar(lvars[i])));
    }
  }
};

template <class A>
class InputOrder : VariableOrder<A> {
public:
  using Allocator = typename A::Allocator;
  using LVarArray = VariableOrder<A>::LVarArray;

private:
  ZDec smallest;

public:
  CUDA InputOrder(InputOrder&&) = default;
  CUDA InputOrder(battery::shared_ptr<A, Allocator> a): VariableOrder(std::move(a)), smallest(ZDec::bot()) {}

  CUDA InputOrder(battery::shared_ptr<A, Allocator> a, const LVarArray& lvars): VariableOrder(std::move(a)), smallest(ZDec::bot()) {}

  CUDA int num_refinements() {
    return vars.size();
  }

  CUDA void restore() {
    smallest.dtell(ZDec::bot());
  }

  CUDA void refine(int i, BInc& has_changed) {
    if(i < vars.size()) {
      const auto& x = a->project(vars[i]);
      // This condition is actually monotone under the assumption that x is not updated anymore between two invocations of this refine function.
      if(lt<A::Universe>(x.lb(), x.ub()).value()) {
        smallest.tell(ZDec(i), has_changed);
      }
    }
  }

  CUDA thrust::optional<AVar> project() const {
    if(smallest.is_bot()) {
      return {};
    }
    return vars[smallest.value()];
  }
};

}

#endif
