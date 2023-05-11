// Copyright 2022 Pierre Talbot

#ifndef BAB_HPP
#define BAB_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/copy_dag_helper.hpp"

namespace lala {

template <class A>
class BAB {
public:
  using allocator_type = typename A::allocator_type;
  using sub_type = A;
  using sub_ptr = battery::shared_ptr<A, allocator_type>;
  using this_type = BAB<A>;

  template <class Alloc2>
  struct tell_type {
    using sub_tell_type = typename sub_type::tell_type<Alloc2>;
    AVar x;
    bool optimization_mode;
    battery::vector<sub_tell_type, Alloc2> sub_tells;
    tell_type() = default;
    tell_type(tell_type&&) = default;
    tell_type(const tell_type&) = default;
    CUDA tell_type(AVar x, bool opt): x(x), optimization_mode(opt) {}
  };

  template <class Alloc2>
  using ask_type = battery::vector<typename sub_type::ask_type<Alloc2>, Alloc2>;

  template<class F, class Env>
  using iresult_tell = IResult<tell_type<typename Env::allocator_type>, F>;

  template<class F, class Env>
  using iresult_ask = IResult<ask_type<typename Env::allocator_type>, F>;

  constexpr static const char* name = "BAB";

  template <class A2>
  friend class BAB;

private:
  AType atype;
  sub_ptr sub;
  sub_ptr best;
  AVar x;
  bool optimization_mode; // `true` for minimization, `false` for maximization.
  int solutions_found;

public:
  CUDA BAB(AType atype, sub_ptr sub)
   : atype(atype), sub(std::move(sub)), x(),
     solutions_found(0)
  {
    assert(this->sub);
    auto deps = AbstractDeps<allocator_type>(this->sub->get_allocator());
    best = deps.template clone<sub_type>(this->sub);
    assert(this->best);
  }

  template<class A2, class FastAlloc>
  CUDA BAB(const BAB<A2>& other, AbstractDeps<allocator_type, FastAlloc>& deps)
   : atype(other.atype), sub(deps.template clone<A>(other.sub)),
     best(deps.template clone<A>(other.best)), x(other.x), optimization_mode(other.optimization_mode)
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return sub->get_allocator();
  }

  CUDA local::BInc is_top() const {
    return sub->is_top();
  }

  CUDA local::BDec is_bot() const {
    return x.is_untyped() && sub->is_bot();
  }

private:
  template <bool is_tell, class R, class SubR, class F, class Env>
  CUDA void interpret_sub(R& res, SubR& sub_res, const F& f, Env& env) {
    if(sub_res.has_value()) {
      if constexpr(is_tell) {
        res.value().sub_tells.push_back(std::move(sub_res.value()));
      }
      else {
        res.value().push_back(std::move(sub_res.value()));
      }
      res.join_warnings(std::move(sub_res));
    }
    else {
      res.join_errors(std::move(sub_res));
    }
  }

  template <bool is_tell, class R, class F, class Env>
  CUDA void interpret_sub(R& res, const F& f, Env& env) {
    if constexpr(is_tell){
      auto sub_res = sub->interpret_tell_in(f, env);
      interpret_sub<is_tell>(res, sub_res, f, env);
    }
    else {
      auto sub_res = sub->interpret_ask_in(f, env);
      interpret_sub<is_tell>(res, sub_res, f, env);
    }
  }

  template <class F, class Env>
  CUDA void interpret_optimization_predicate(iresult_tell<F, Env>& res, const F& f, Env& env) {
    if(f.is_untyped() || f.type() == aty()) {
      if(f.is(F::Seq) && (f.sig() == MAXIMIZE || f.sig() == MINIMIZE)) {
        res.value().optimization_mode = f.sig() == MINIMIZE;
        if(f.seq(0).is_variable()) {
          auto var_res = env.interpret(f.seq(0));
          if(var_res.has_value()) {
            res.value().x = var_res.value();
          }
          else {
            res.join_errors(std::move(var_res));
          }
        }
        else {
          res = iresult_tell<F, Env>(IError<F>(true, name, "Optimization predicates expect a variable to optimize. Instead, you can create a new variable with the expression to optimize.", f));
        }
        return;
      }
      else if(f.type() == aty()) {
        res = iresult_tell<F, Env>(IError<F>(true, name, "Unsupported formula.", f));
        return;
      }
    }
    interpret_sub<true>(res, f, env);
  }

  template <class R, class F, class Env>
  CUDA void interpret_tell_in(R& res, const F& f, Env& env) {
    if(f.is_untyped() || f.type() == aty()) {
      if(f.is(F::Seq) && f.sig() == AND) {
        for(int i = 0; i < f.seq().size(); ++i) {
          interpret_tell_in(res, f.seq(i), env);
        }
      }
      else {
        interpret_optimization_predicate(res, f, env);
      }
    }
    else {
      interpret_sub<true>(res, f, env);
    }
  }

public:
  template <class F, class Env>
  CUDA iresult_tell<F, Env> interpret_tell_in(const F& f, Env& env) {
    iresult_tell<F, Env> res(tell_type<typename Env::allocator_type>{});
    interpret_tell_in(res, f, env);
    return std::move(res);
  }

  template <class F, class Env>
  CUDA iresult_ask<F, Env> interpret_ask_in(const F& f, Env& env) {
    iresult_ask<F, Env> res{env.get_allocator()};
    interpret_sub<false>(res, f, env);
    return std::move(res);
  }

  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    for(int i = 0; i < t.sub_tells.size(); ++i) {
      sub->tell(t.sub_tells[i], has_changed);
    }
    if(!t.x.is_untyped()) {
      assert(x.is_untyped()); // multi-objective optimization not yet supported.
      x = t.x;
      optimization_mode = t.optimization_mode;
      has_changed.tell_top();
    }
    return *this;
  }

  /** \return `true` if the sub-domain is a solution (more precisely, an under-approximation) of the problem.
      The extracted under-approximation can be retreived by `optimum()`. */
  template <class Mem>
  CUDA bool refine(BInc<Mem>& has_changed) {
    bool found_solution = sub->extract(*best);
    if(!x.is_untyped() && found_solution) {
      solutions_found++;
      Sig optimize_sig = is_minimization() ? LT : GT;
      typename sub_type::universe_type k = is_minimization()
        ? best->project(x).lb()
        : best->project(x).ub();
      using F = TFormula<allocator_type>;
      VarEnv<allocator_type> empty_env{};
      F constant = k.template deinterpret<F>();
      auto opti_fun = F::make_binary(F::make_avar(x), optimize_sig, constant, UNTYPED, get_allocator());
      auto t = sub->interpret_tell_in(opti_fun, empty_env).value();
      sub->tell(t, has_changed);
    }
    return found_solution;
  }

  CUDA int solutions_count() const {
    return solutions_found;
  }

  /** An under-approximation is reached when the underlying abstract element `sub` is equal to `top`.
   * We consider that `top` implies we have completely explored `sub`, and we can't find better bounds.
   * As this abstract element cannot further refine `best`, it is shared with the under-approximation.
   * It is safe to use `this.extract(*this)` to avoid allocating memory. */
  template <class A2>
  CUDA bool extract(BAB<A2>& ua) const {
    if(solutions_found > 0 && sub->is_top())
    {
      ua.solutions_found = solutions_found;
      best->extract(*(ua.best));
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

  CUDA AVar objective_var() const {
    return x;
  }
};

}

#endif
