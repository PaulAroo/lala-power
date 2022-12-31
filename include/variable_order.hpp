// Copyright 2022 Pierre Talbot

#ifndef VARIABLE_ORDER_HPP
#define VARIABLE_ORDER_HPP

#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

namespace lala {

template <class A>
class VariableOrder
{
public:
  using allocator_type = typename A::allocator_type;
  template <class Alloc>
  using tell_type = typename A::tell_type<Alloc>;
  using LVarArray = battery::vector<LVar<allocator_type>, allocator_type>;

protected:
  battery::vector<AVar, allocator_type> vars;
  battery::shared_ptr<A, allocator_type> a;

public:
  VariableOrder(VariableOrder&&) = default;
  CUDA VariableOrder(battery::shared_ptr<A, allocator_type> a) : a(a) {}

  template<class A2>
  CUDA VariableOrder(const VariableOrder<A2>& other, AbstractDeps<allocator_type>& deps)
   : vars(other.vars), a(deps.clone(other.a)) {}

  template <class Env>
  CUDA void interpret_in(const Env& env) {
    if(vars.size() != env.num_vars()) {
      vars.clear();
      vars.reserve(env.num_vars());
      for(int i = 0; i < env.num_vars(); ++i) {
        vars.push_back(env[i].avars[0]); // We suppose the first recorded abstract variable is the most general.
      }
    }
  }

  CUDA local::BInc is_top() const {
    return a->is_top();
  }
};

template <class A>
class InputOrder : public VariableOrder<A> {
public:
  using allocator_type = typename A::allocator_type;
  using universe_type = typename A::universe_type;
  using memory_type = typename universe_type::memory_type;
  using LVarArray = typename VariableOrder<A>::LVarArray;

private:
  ZDec<int, memory_type> smallest;

public:
  InputOrder(InputOrder&&) = default;
  CUDA InputOrder(battery::shared_ptr<A, allocator_type> a)
   : VariableOrder<A>(std::move(a)) {}
  CUDA InputOrder(battery::shared_ptr<A, allocator_type> a, const LVarArray& lvars)
   : VariableOrder<A>(std::move(a)) {}

  template<class A2>
  CUDA InputOrder(const InputOrder<A2>& other, AbstractDeps<allocator_type>& deps)
   : VariableOrder<A>(other, deps), smallest(other.smallest) {}

  CUDA int num_refinements() const {
    return this->vars.size();
  }

  CUDA void reset() {
    smallest.dtell_bot();
  }

  template <class Mem>
  CUDA void refine(int i, BInc<Mem>& has_changed) {
    using LB = typename universe_type::LB;
    static_assert(LB::preserve_inner_covers); // a way to restrict this function to discrete domain.
    const universe_type& x = this->a->project(this->vars[i]);
    if(x.lb() < dual<LB>(x.ub())) {
      smallest.tell(ZDec<int, memory_type>(i), has_changed);
    }
  }

  CUDA thrust::optional<AVar> project() const {
    if(smallest.is_bot()) {
      return {};
    }
    return this->vars[smallest.value()];
  }
};

}

#endif
