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

protected:
  battery::vector<AVar, Allocator> vars;
  battery::shared_ptr<A, Allocator> a;

public:
  CUDA VariableOrder(VariableOrder&&) = default;
  CUDA VariableOrder(battery::shared_ptr<A, Allocator> a) : a(a) {
    const auto& env = a->environment();
    vars.reserve(env.size());
    for(int i = 0; i < env.size(); ++i) {
      vars.push_back(make_var(a->uid(), i));
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
  ZDec<int> smallest;

public:
  CUDA InputOrder(InputOrder&&) = default;
  CUDA InputOrder(battery::shared_ptr<A, Allocator> a): VariableOrder<A>(std::move(a)), smallest(ZDec<int>::bot()) {}

  CUDA InputOrder(battery::shared_ptr<A, Allocator> a, const LVarArray& lvars): VariableOrder<A>(std::move(a)), smallest(ZDec<int>::bot()) {}

  CUDA int num_refinements() const {
    return this->vars.size();
  }

  CUDA void reset() {
    smallest.dtell(ZDec<int>::bot());
  }

  CUDA void refine(int i, BInc& has_changed) {
    if(i < this->vars.size()) {
      using D = A::Universe;
      const D& x = this->a->project(this->vars[i]);
      // This condition is actually monotone under the assumption that x is not updated anymore between two invocations of this refine function.
      if(lt<typename D::LB>(x.lb(), x.ub()).value()) {
        smallest.tell(ZDec<int>(i), has_changed);
      }
    }
  }

  CUDA thrust::optional<AVar> project() const {
    if(smallest.is_bot().value()) {
      return {};
    }
    return this->vars[smallest.value()];
  }
};

}

#endif
