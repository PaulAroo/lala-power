// Copyright 2022 Pierre Talbot

#ifndef BAB_HPP
#define BAB_HPP

#include "ast.hpp"
#include "z.hpp"
#include "arithmetic.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

namespace lala {

template <class A>
class BAB {
public:
  using Allocator = typename A::Allocator;
  using APtr = battery::shared_ptr<A, Allocator>;
  using this_type = BAB<A>;
  using Env = typename A::Env;

private:
  AType uid_;
  APtr a;
  APtr best;
  AVar x;
  bool optimization_mode; // `true` for minimization, `false` for maximization.
  ZPInc<int> solutions_found;

public:
  CUDA BAB(AType uid, APtr a)
   : uid_(uid), a(std::move(a)), x(-1),
     solutions_found(0),
     best(AbstractDeps<Allocator>(this->a->get_allocator()).clone(this->a))
  {
    assert(this->a);
    assert(this->best);
  }

  template<class A2>
  CUDA BAB(const BAB<A2>& other, AbstractDeps<Allocator>& deps)
   : uid_(other.uid_), a(deps.clone(other.a)),
     best(deps.clone(other.best)), x(other.x), optimization_mode(other.optimization_mode)
  {}

  CUDA AType uid() const {
    return uid_;
  }

  CUDA Allocator get_allocator() const {
    return a->get_allocator();
  }

  CUDA BInc is_top() const {
    return a->is_top();
  }

  CUDA BDec is_bot() const {
    return x == -1 && a->is_bot().value();
  }

  struct TellType {
    AVar x;
    bool optimization_mode;
    typename A::TellType a_tell;
    TellType(AVar x, bool opt, typename A::TellType&& t)
     : x(x), optimization_mode(opt), a_tell(std::move(t)) {}
  };

  template <class F>
  CUDA thrust::optional<TellType> interpret(const F& f) {
    if(f.mode() != F::SATISFY) {
      auto a_tell = a->interpret(f.formula());
      if(a_tell.has_value()) {
        auto x = a->environment().to_avar(f.optimization_lvar());
        if(x.has_value()) {
          return TellType(*x, f.mode() == F::MINIMIZE, std::move(*a_tell));
        }
      }
    }
    return {};
  }

  CUDA this_type& tell(TellType&& t, BInc& has_changed) {
    assert(x == -1); // multi-objective optimization not yet supported.
    a->tell(std::move(t.a_tell), has_changed);
    x = t.x;
    optimization_mode = t.optimization_mode;
    return *this;
  }

  CUDA void refine(BInc& has_changed) {
    if(x != -1 && a->extract(*best)) {
      solutions_found.tell(add(solutions_found, spos(1)));
      Sig optimize_sig = is_minimization() ? LT : GT;
      auto k = is_minimization()
        ? best->project(x).lb().value()
        : best->project(x).ub().value();
      using F = TFormula<Allocator>;
      auto t = *(a->interpret(F::make_binary(F::make_avar(x), optimize_sig, F::make_z(k), UNTYPED, EXACT, get_allocator())));
      a->tell(std::move(t), has_changed);
    }
  }

  CUDA ZPInc<int> solutions_count() const {
    return solutions_found;
  }

  /** An under-approximation is reached when the underlying abstract element `a` is equal to `top`.
   * We consider that `top` implies we have completely explored `a`, and we can't find better bounds.
   * As this abstract element cannot further refine `best`, it is shared with the under-approximation.
   * It is safe to use `this.extract(*this)` to avoid allocating memory. */
  CUDA bool extract(BAB<A>& ua) const {
    if(land(gt<ZPInc<int>>(solutions_found, 0),
            a->is_top()).guard())
    {
      ua.solutions_found = ZPInc<int>(solutions_found);
      ua.best = best;
      ua.a = a;
      ua.x = x;
      ua.optimization_mode = optimization_mode;
      return true;
    }
    return false;
  }

  /** \pre `extract` must return `true`, otherwise it might not be an optimum. */
  CUDA const A& optimum() const {
    return *best;
  }

  CUDA bool is_minimization() const {
    return optimization_mode;
  }

  CUDA bool is_maximization() const {
    return !optimization_mode;
  }

  CUDA const Env& environment() const {
    if(!a->is_top().guard()) {
      return a->environment();
    }
    else {
      return best->environment();
    }
  }
};

}

#endif
