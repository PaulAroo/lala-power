// Copyright 2023 Pierre Talbot

#ifndef SPLIT_STRATEGY_HPP
#define SPLIT_STRATEGY_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "branch.hpp"
#include "lala/logic/logic.hpp"

namespace lala {

enum class VariableOrder {
  INPUT_ORDER,
  FIRST_FAIL,
  ANTI_FIRST_FAIL,
  SMALLEST,
  LARGEST,
  // unsupported:
  // OCCURRENCE,
  // MOST_CONSTRAINED,
  // MAX_REGRET,
  // DOM_W_DEG,
  // RANDOM
};

enum class ValueOrder {
  MIN,
  MAX,
  MEDIAN,
  SPLIT,
  REVERSE_SPLIT,
  // unsupported:
  // INTERVAL,
  // RANDOM,
  // MIDDLE,
};

namespace impl {
  template <class U>
  auto median(const U& u) -> decltype(u.median()) {
    return u.median();
  }

  template <class U>
  U median(const U& u) {
    printf("error: this universe of discourse has no function `median`, please change the search strategy.");
    assert(false);
    return u;
  }
}

template <class A>
class SplitStrategy {
public:
  using allocator_type = typename A::allocator_type;
  using tell_type = typename A::tell_type<allocator_type>;
  using branch_type = Branch<tell_type, allocator_type>;

  template <class>
  using snapshot_type = battery::tuple<int, int>;

  /** A split strategy consists of a variable order and value order on a subset of the variables. */
  template <class Alloc2>
  using strategy_type = battery::tuple<
    VariableOrder,
    ValueOrder,
    battery::vector<AVar, Alloc2>>;

  template<class F, class Env>
  using iresult_tell = IResult<bool, F>;

  constexpr static const char* name = "SplitStrategy";

private:
  using universe_type = typename A::universe_type;
  using LB = typename universe_type::LB;
  using UB = typename universe_type::UB;

  AType atype;
  battery::shared_ptr<A, allocator_type> a;
  battery::vector<strategy_type<allocator_type>, allocator_type> strategies;
  int current_strategy;
  int next_unassigned_var;

  CUDA const battery::vector<AVar, allocator_type>& current_vars() const {
    return battery::get<2>(strategies[current_strategy]);
  }

  CUDA void move_to_next_unassigned_var() {
    while(current_strategy < strategies.size()) {
      const auto& vars = current_vars();
      while(next_unassigned_var < vars.size()) {
        const auto& v = a.project(vars[next_unassigned_var]);
        if(v.lb() < dual<A::LB>(v.ub())) {
          return;
        }
        next_unassigned_var++;
      }
      current_strategy++;
      next_unassigned_var = 0;
    }
  }

  template <class MapFunction>
  CUDA AVar var_map_fold_left(const battery::vector<AVar, allocator_type>& vars, MapFunction op) {
    int i = next_unassigned_var;
    int best_i = i;
    auto best = op(a->project(vars[i]));
    for(++i; i < vars.size(); ++i) {
      const auto& u = a->project(vars[i]);
      if(u.lb() < dual<LB>(u.ub())) {
        local::BInc has_changed;
        best.tell(op(u), has_changed);
        if(has_changed) {
          best_i = i;
        }
      }
    }
    return vars[best_i];
  }

  CUDA AVar select_var() {
    const auto& strat = strategies[current_strategy];
    const auto& vars = battery::get<2>(strat);
    switch(battery::get<0>(strat)) {
      case VariableOrder::INPUT_ORDER: return battery::get<2>(vars)[next_unassigned_var];
      case VariableOrder::FIRST_FAIL: return var_map_fold_left(vars, [](const universe_type& u) { return u.width(); });
      case VariableOrder::ANTI_FIRST_FAIL: return anti_first_fail(vars, [](const universe_type& u) { return dual<local::ZInc>(u.width()); });
      case VariableOrder::LARGEST: return largest(vars, [](const universe_type& u) { return dual<LB>(u.ub()); });
      case VariableOrder::SMALLEST: return smallest(vars, [](const universe_type& u) { return dual<UB>(u.lb()); });
    }
  }

  template <class Env, class U>
  CUDA branch_type make_branch(Env& env, AVar x, Sig left, Sig right, const U& u) {
    if(u.is_top() && U::preserved_top || u.is_bot() && U::preserved_bot) {
      return branch_type{};
    }
    using F = TFormula<typename Env::allocator_type>;
    using branch_vector = battery::vector<tell_type, allocator_type>;
    auto k = U::value_type::deinterpret(u.value(), env.get_allocator());
    auto left = a->interpret_tell_in(F::make_binary(F::make_avar(x), left, k, UNTYPED, env.get_allocator()), env);
    auto right = a->interpret_tell_in(F::make_binary(F::make_avar(x), right, k, UNTYPED, env.get_allocator()), env);
    if(left.has_value() && right.has_value()) {
      return Branch(branch_vector({std::move(left.value()), std::move(right.value())}, a->get_allocator()));
    }
    else {
      left.print_diagnostics();
      right.print_diagnostics();
      return branch_type{};
    }
  }

public:
  CUDA SplitStrategy(AType atype, battery::shared_ptr<A, allocator_type> a):
    atype(atype), a(a), current_strategy(0), next_unassigned_var(0) {}

  template<class A2, class FastAlloc>
  CUDA SplitStrategy(const SplitStrategy<A2>& other, AbstractDeps<allocator_type, FastAlloc>& deps)
   : atype(other.atype),
     a(deps.template clone<A>(other.a)),
     strategies(other.strategies, deps.get_allocator()),
     current_strategy(other.current_strategy),
     next_unassigned_var(other.next_unassigned_var)
  {}

  CUDA AType aty() const {
    return atype;
  }

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot() const {
    return battery::make_tuple<int, int>(current_strategy, next_unassigned_var);
  }

  template <class Alloc2 = allocator_type>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    current_strategy = battery::get<0>(snap);
    next_unassigned_var = battery::get<1>(snap);
  }

  /** This interpretation function expects `f` to be a predicate of the form `search(VariableOrder, ValueOrder, x_1, x_2, ..., x_n)`.
   * Calling this function multiple times will add multiple strategies, that will be called in sequence along a branch of the search tree.
   * In case of success, the interpreted strategy is added in the strategies, without the need to call a subsequent `tell` operation (rational: `tell` is designed to be compatible with parallel operations, but adding a strategy must be done sequentially in this domain). */
  template <class F, class Env>
  CUDA iresult_tell<F, Env> interpret_tell_in(const F& f, Env& env) {
    if(f.is(F::ESeq)
      && f.seq().size() >= 3
      && f.esig() == "search"
      && f.seq()[0].is(F::ESeq) && f.seq()[0].size() == 0
      && f.seq()[1].is(F::ESeq) && f.seq()[1].size() == 0)
    {
      return iresult_tell<F, Env>(
        IError<F>(true, name,
          "We only interpret predicate of the form `search(input_order, indomain_min, x1, ..., xN)`.", f));
    }
    VariableOrder var_order;
    ValueOrder val_order;
    const auto& var_order_str = f.seq()[0].esig();
    const auto& val_order_str = f.seq()[1].esig();
    if(var_order_str == "input_order") { var_order = VariableOrder::INPUT_ORDER; }
    else if(var_order_str == "first_fail") { var_order = VariableOrder::FIRST_FAIL; }
    else if(var_order_str == "anti_first_fail") { var_order = VariableOrder::ANTI_FIRST_FAIL; }
    else if(var_order_str == "smallest") { var_order = VariableOrder::SMALLEST; }
    else if(var_order_str == "largest") { var_order = VariableOrder::LARGEST; }
    else {
      return iresult_tell<F, Env>(
        IError<F>(true, name, "This variable order strategy is unsupported.", f));
    }
    if(val_order_str == "indomain_min") { var_order = ValueOrder::MIN; }
    else if(val_order_str == "indomain_max") { var_order = ValueOrder::MAX; }
    else if(val_order_str == "indomain_median") { var_order = ValueOrder::MEDIAN; }
    else if(val_order_str == "indomain_split") { var_order = ValueOrder::SPLIT; }
    else if(val_order_str == "indomain_reverse_split") { var_order = ValueOrder::REVERSE_SPLIT; }
    else {
      return iresult_tell<F, Env>(
        IError<F>(true, name, "This value order strategy is unsupported.", f));
    }
    battery::vector<AVar, typename Env::allocator_type> vars;
    for(int i = 2; i < f.seq().size(); ++i) {
      if(f.seq(i).is(LV)) {
        auto res_var = env.interpret_lv(f.seq(i).lv());
        if(res_var.has_value()) {
          vars.push_back(res_var.value());
        }
        else {
          return iresult_tell<F, Env>(res_var.error());
        }
      }
      else if(f.seq(i).is(V)) {
        vars.push_back(f.seq(i).v());
      }
      else {
        return iresult_tell<F, Env>(
          IError<F>(true, name, "A non-variable expression is passed to the predicate `search` after the variable and value order strategies.", f.seq(i)));
      }
    }
    strategies.push_back(battery::make_tuple(var_order, val_order, std::move(vars)));
    return iresult_tell<F, Env>(true);
  }

  /** Split the next unassigned variable according to the current strategy.
   * If all variables of the current strategy are assigned, use the next strategy.
   * If no strategy remains, returns an empty set of branches.

   If the next unassigned variable cannot be split, for instance because the value ordering strategy maps to `bot` or `top`, an empty set of branches is returned.
   This also means that you cannot suppose `split(a) = {}` to mean `a` is at `top`. */
  template <class Env>
  CUDA branch_type split(Env& env) {
    move_to_next_unassigned_var();
    if(current_strategy < strategies.size()) {
      AVar x = select_var(env);
      switch(battery::get<1>(strategies[current_strategy])) {
        case ValueOrder::MIN: return make_branch(x, EQ, NEQ, a->project(x).lb());
        case ValueOrder::MAX: return make_branch(x, EQ, NEQ, a->project(x).ub());
        case ValueOrder::MEDIAN: return make_branch(x, EQ, NEQ, impl::median(a->project(x)));
        case ValueOrder::SPLIT: return make_branch(x, LEQ, GT, impl::median(a->project(x)));
        case ValueOrder::REVERSE_SPLIT: return make_branch(x, GT, LEQ, impl::median(a->project(x)));
      }
    }
    else {
      return branch_type{};
    }
  }
};

}

#endif