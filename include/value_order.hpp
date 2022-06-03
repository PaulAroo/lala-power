// Copyright 2022 Pierre Talbot

#ifndef VALUE_ORDER_HPP
#define VALUE_ORDER_HPP

#include "ast.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"
#include "branch.hpp"

namespace lala {

template <class A, Approx appx = EXACT>
class LowerBound
{
public:
  using Allocator = typename A::Allocator;
  using TellType = typename A::TellType;
  using BranchType = Branch<TellType, Allocator>;

private:
  battery::shared_ptr<A, Allocator> a;

public:
  LowerBound(LowerBound&&) = default;
  CUDA LowerBound(battery::shared_ptr<A, Allocator> a)
   : a(std::move(a)) {}

  template<class A2>
  CUDA LowerBound(const LowerBound<A2, appx>& other, AbstractDeps<Allocator>& deps)
   : a(deps.clone(other.a)) {}

  /** Create two formulas of the form `x = lb \/ x > lb`.
   *  We suppose that `a` is able to interpret those constraints. */
  CUDA BranchType split(AVar x) const {
    using F = TFormula<Allocator>;
    auto lb = a->project(x).lb().value();
    return Branch(battery::vector<TellType, Allocator>({
        *(a->interpret(F::make_binary(F::make_avar(x), EQ, F::make_z(lb), UNTYPED, appx, a->get_allocator()))),
        *(a->interpret(F::make_binary(F::make_avar(x), GT, F::make_z(lb), UNTYPED, appx, a->get_allocator())))
      },
      a->get_allocator()));
  }
};

}

#endif
