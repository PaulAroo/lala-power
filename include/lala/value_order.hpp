// Copyright 2022 Pierre Talbot

#ifndef VALUE_ORDER_HPP
#define VALUE_ORDER_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "branch.hpp"
#include "lala/logic/logic.hpp"

namespace lala {

template <class A, Approx appx = EXACT>
class LowerBound
{
public:
  using allocator_type = typename A::allocator_type;
  template <class Alloc>
  using tell_type = typename A::tell_type<Alloc>;
  using branch_type = Branch<tell_type<allocator_type>, allocator_type>;

private:
  battery::shared_ptr<A, allocator_type> a;

public:
  LowerBound(LowerBound&&) = default;
  CUDA LowerBound(battery::shared_ptr<A, allocator_type> a)
   : a(std::move(a)) {}

  template<class A2, class FastAlloc>
  CUDA LowerBound(const LowerBound<A2, appx>& other, AbstractDeps<allocator_type, FastAlloc>& deps)
   : a(deps.template clone<A>(other.a)) {}

  /** Create two formulas of the form `x = lb \/ x > lb`.
   *  We suppose that `a` is able to interpret those constraints. */
  template <class Env>
  CUDA branch_type split(AVar x, Env& env) const {
    using F = TFormula<allocator_type>;
    auto lb = a->project(x).lb();
    return Branch(battery::vector<tell_type<allocator_type>, allocator_type>({
        std::move(a->interpret_in(F::make_binary(F::make_avar(x), EQ, F::make_z(lb), UNTYPED, appx, a->get_allocator()), env).value()),
        std::move(a->interpret_in(F::make_binary(F::make_avar(x), GT, F::make_z(lb), UNTYPED, appx, a->get_allocator()), env).value())
      },
      a->get_allocator()));
  }
};

}

#endif
