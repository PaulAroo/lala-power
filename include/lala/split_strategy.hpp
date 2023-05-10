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

template <class A>
class SplitStrategy {
public:
  using allocator_type = typename A::allocator_type;
  using sub_tell_type = typename A::tell_type<allocator_type>;
  using branch_type = Branch<sub_tell_type, allocator_type>;
  using this_type = SplitStrategy<A>;
  template <class>
  using snapshot_type = battery::tuple<int, int, int>;

  /** A split strategy consists of a variable order and value order on a subset of the variables. */
  template <class Alloc2>
  using strategy_type = battery::tuple<
    VariableOrder,
    ValueOrder,
    battery::vector<AVar, Alloc2>>;

  template <class Alloc2>
  using tell_type = strategy_type<Alloc2>;

  template<class F, class Env>
  using iresult_tell = IResult<tell_type<typename Env::allocator_type>, F>;

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
        const auto& v = a->project(vars[next_unassigned_var]);
        if(v.lb() < dual<LB>(v.ub())) {
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
      case VariableOrder::INPUT_ORDER: return vars[next_unassigned_var];
      case VariableOrder::FIRST_FAIL: return var_map_fold_left(vars, [](const universe_type& u) { return u.width().ub(); });
      case VariableOrder::ANTI_FIRST_FAIL: return var_map_fold_left(vars, [](const universe_type& u) { return dual<LB>(u.width().ub()); });
      case VariableOrder::LARGEST: return var_map_fold_left(vars, [](const universe_type& u) { return dual<LB>(u.ub()); });
      case VariableOrder::SMALLEST: return var_map_fold_left(vars, [](const universe_type& u) { return dual<UB>(u.lb()); });
      default: printf("unsupported variable order strategy\n"); assert(false); return AVar{};
    }
  }

  template <class U>
  CUDA branch_type make_branch(AVar x, Sig left_op, Sig right_op, const U& u) {
    if(u.is_top() && U::preserve_top || u.is_bot() && U::preserve_bot) {
      return branch_type{};
    }
    using F = TFormula<allocator_type>;
    using branch_vector = battery::vector<sub_tell_type, allocator_type>;
    VarEnv<allocator_type> empty_env{};
    auto k = u.template deinterpret<F>();
    auto left = a->interpret_tell_in(F::make_binary(F::make_avar(x), left_op, k, UNTYPED, a->get_allocator()), empty_env);
    auto right = a->interpret_tell_in(F::make_binary(F::make_avar(x), right_op, k, UNTYPED, a->get_allocator()), empty_env);
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
    return battery::make_tuple((int)strategies.size(), current_strategy, next_unassigned_var);
  }

  template <class Alloc2 = allocator_type>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    while(strategies.size() > battery::get<0>(snap)) {
      strategies.pop_back();
    }
    current_strategy = battery::get<1>(snap);
    next_unassigned_var = battery::get<2>(snap);
  }

  /** This interpretation function expects `f` to be a predicate of the form `search(VariableOrder, ValueOrder, x_1, x_2, ..., x_n)`. */
  template <class F, class Env>
  CUDA iresult_tell<F, Env> interpret_tell_in(const F& f, Env& env) {
    if(!(f.is(F::ESeq)
      && f.eseq().size() >= 3
      && f.esig() == "search"
      && f.eseq()[0].is(F::ESeq) && f.eseq()[0].eseq().size() == 0
      && f.eseq()[1].is(F::ESeq) && f.eseq()[1].eseq().size() == 0))
    {
      return iresult_tell<F, Env>(
        IError<F>(true, name,
          "We only interpret predicate of the form `search(input_order, indomain_min, x1, ..., xN)`.", f));
    }
    VariableOrder var_order;
    ValueOrder val_order;
    const auto& var_order_str = f.eseq()[0].esig();
    const auto& val_order_str = f.eseq()[1].esig();
    if(var_order_str == "input_order") { var_order = VariableOrder::INPUT_ORDER; }
    else if(var_order_str == "first_fail") { var_order = VariableOrder::FIRST_FAIL; }
    else if(var_order_str == "anti_first_fail") { var_order = VariableOrder::ANTI_FIRST_FAIL; }
    else if(var_order_str == "smallest") { var_order = VariableOrder::SMALLEST; }
    else if(var_order_str == "largest") { var_order = VariableOrder::LARGEST; }
    else {
      return iresult_tell<F, Env>(
        IError<F>(true, name, "This variable order strategy is unsupported.", f));
    }
    if(val_order_str == "indomain_min") { val_order = ValueOrder::MIN; }
    else if(val_order_str == "indomain_max") { val_order = ValueOrder::MAX; }
    else if(val_order_str == "indomain_median") { val_order = ValueOrder::MEDIAN; }
    else if(val_order_str == "indomain_split") { val_order = ValueOrder::SPLIT; }
    else if(val_order_str == "indomain_reverse_split") { val_order = ValueOrder::REVERSE_SPLIT; }
    else {
      return iresult_tell<F, Env>(
        IError<F>(true, name, "This value order strategy is unsupported.", f));
    }
    battery::vector<AVar, typename Env::allocator_type> vars;
    for(int i = 2; i < f.eseq().size(); ++i) {
      if(f.eseq(i).is(F::LV)) {
        auto res_var = env.interpret(f.eseq(i));
        if(res_var.has_value()) {
          vars.push_back(res_var.value());
        }
        else {
          return std::move(res_var.error());
        }
      }
      else if(f.eseq(i).is(F::V)) {
        vars.push_back(f.eseq(i).v());
      }
      else {
        return iresult_tell<F, Env>(
          IError<F>(true, name, "A non-variable expression is passed to the predicate `search` after the variable and value order strategies.", f.eseq(i)));
      }
    }
    return iresult_tell<F, Env>(battery::make_tuple(var_order, val_order, std::move(vars)));
  }

  template <class Alloc2>
  CUDA this_type& tell(const tell_type<Alloc2>& t) {
    strategies.push_back(t);
    return *this;
  }

  /** Calling this function multiple times will create multiple strategies, that will be called in sequence along a branch of the search tree. */
  template <class Alloc2, class Mem>
  CUDA this_type& tell(const tell_type<Alloc2>& t, BInc<Mem>& has_changed) {
    has_changed.tell_top();
    return tell(t);
  }

  /** Split the next unassigned variable according to the current strategy.
   * If all variables of the current strategy are assigned, use the next strategy.
   * If no strategy remains, returns an empty set of branches.

   If the next unassigned variable cannot be split, for instance because the value ordering strategy maps to `bot` or `top`, an empty set of branches is returned.
   This also means that you cannot suppose `split(a) = {}` to mean `a` is at `top`. */
  CUDA branch_type split() {
    move_to_next_unassigned_var();
    if(current_strategy < strategies.size()) {
      AVar x = select_var();
      switch(battery::get<1>(strategies[current_strategy])) {
        case ValueOrder::MIN: return make_branch(x, EQ, GT, a->project(x).lb());
        case ValueOrder::MAX: return make_branch(x, EQ, LT, a->project(x).ub());
        case ValueOrder::MEDIAN: return make_branch(x, EQ, NEQ, a->project(x).median().lb());
        case ValueOrder::SPLIT: return make_branch(x, LEQ, GT, a->project(x).median().lb());
        case ValueOrder::REVERSE_SPLIT: return make_branch(x, GT, LEQ, a->project(x).median().lb());
        default: printf("unsupported value order strategy\n"); assert(false); return branch_type{};
      }
    }
    else {
      return branch_type{};
    }
  }
};

}

#endif