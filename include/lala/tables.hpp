// Copyright 2023 Pierre Talbot

#ifndef LALA_POWER_TABLES_HPP
#define LALA_POWER_TABLES_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/dynamic_bitset.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"

namespace lala {

template <class A, class U, class Alloc> class Tables;
namespace impl {
  template <class>
  struct is_table_like {
    static constexpr bool value = false;
  };
  template<class A, class U, class Alloc>
  struct is_table_like<Tables<A, U, Alloc>> {
    static constexpr bool value = true;
  };
}

/** The table abstract domain is designed to represent predicates in extension by listing all their solutions explicitly.
 * It is inspired by the table global constraint and generalizes it by lifting each element of the table to a lattice element.
 * We expect `U` to be equally or less expressive than `A::universe_type`, this is because we compute the meet in `A::universe_type` and not in `U`.
 */
template <class A, class U = typename A::universe_type, class Allocator = typename A::allocator_type>
class Tables {
public:
  using allocator_type = Allocator;
  using sub_allocator_type = typename A::allocator_type;
  using universe_type = U;
  using local_universe = typename universe_type::local_type;
  using sub_universe_type = typename A::universe_type;
  using sub_local_universe = typename sub_universe_type::local_type;
  using memory_type = typename universe_type::memory_type;
  using sub_type = A;
  using sub_ptr = abstract_ptr<sub_type>;
  using this_type = Tables<sub_type, universe_type, allocator_type>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = sub_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = sub_type::preserve_bot;
  constexpr static const bool preserve_top = sub_type::preserve_top;
  // The next properties should be checked more seriously, relying on the sub-domain might be uneccessarily restrictive.
  constexpr static const bool preserve_join = sub_type::preserve_join;
  constexpr static const bool preserve_meet = sub_type::preserve_meet;
  constexpr static const bool injective_concretization = sub_type::injective_concretization;
  constexpr static const bool preserve_concrete_covers = sub_type::preserve_concrete_covers;
  constexpr static const char* name = "Tables";

  using table_type = battery::vector<
    battery::vector<universe_type, allocator_type>,
    allocator_type>;
  using table_collection_type = battery::vector<table_type, allocator_type>;
  using bitset_type = battery::dynamic_bitset<memory_type, allocator_type>;

private:
  AType atype;
  AType store_aty;
  sub_ptr sub;

  battery::vector<battery::vector<AVar, allocator_type>, allocator_type> headers;
  table_collection_type tell_tables;
  table_collection_type ask_tables;
  battery::vector<bitset_type, allocator_type> eliminated_rows;
  // See `refine`.
  battery::vector<size_t, allocator_type> table_idx_to_column;
  battery::vector<size_t, allocator_type> column_to_table_idx;
  size_t total_cells;

public:
  template <class Alloc>
  struct tell_type {
    using allocator_type = Alloc;

    typename A::template tell_type<Alloc> sub;

    battery::vector<battery::vector<AVar, Alloc>, Alloc> headers;

    battery::vector<battery::vector<
      battery::vector<universe_type, Alloc>,
    Alloc>, Alloc> tell_tables;

    battery::vector<battery::vector<
      battery::vector<universe_type, Alloc>,
    Alloc>, Alloc> ask_tables;

    CUDA tell_type(const Alloc& alloc = Alloc{})
     : sub(alloc)
     , headers(alloc)
     , tell_tables(alloc)
     , ask_tables(alloc)
    {}
    tell_type(const tell_type&) = default;
    tell_type(tell_type&&) = default;
    tell_type& operator=(tell_type&&) = default;
    tell_type& operator=(const tell_type&) = default;

    template <class TableTellType>
    CUDA NI tell_type(const TableTellType& other, const Alloc& alloc = Alloc{})
      : sub(other.sub, alloc)
      , headers(other.headers, alloc)
      , tell_tables(other.tell_tables, alloc)
      , ask_tables(other.ask_tables, alloc)
    {}

    CUDA allocator_type get_allocator() const {
      return headers.get_allocator();
    }

    template <class Alloc2>
    friend class tell_type;
  };

  template <class Alloc>
  struct ask_type {
    using allocator_type = Alloc;

    typename A::template ask_type<Alloc> sub;
    battery::vector<battery::vector<AVar, Alloc>, Alloc> headers;
    battery::vector<battery::vector<
      battery::vector<universe_type, Alloc>,
    Alloc>, Alloc> ask_tables;

    CUDA ask_type(const Alloc& alloc = Alloc{})
     : sub(alloc)
     , headers(alloc)
     , ask_tables(alloc)
    {}
    ask_type(const ask_type&) = default;
    ask_type(ask_type&&) = default;
    ask_type& operator=(ask_type&&) = default;
    ask_type& operator=(const ask_type&) = default;

    template <class TableAskType>
    CUDA NI ask_type(const TableAskType& other, const Alloc& alloc = Alloc{})
      : sub(other.sub, alloc)
      , headers(other.headers, alloc)
      , ask_tables(other.ask_tables, alloc)
    {}

    CUDA allocator_type get_allocator() const {
      return headers.get_allocator();
    }

    template <class Alloc2>
    friend class ask_type;
  };

  template <class A2, class U2, class Alloc2>
  friend class Tables;

public:
  CUDA Tables(AType uid, AType store_aty, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : atype(uid)
   , store_aty(store_aty)
   , sub(std::move(sub))
   , headers(alloc)
   , tell_tables(alloc)
   , ask_tables(alloc)
   , eliminated_rows(alloc)
   , table_idx_to_column({0}, alloc)
   , column_to_table_idx(alloc)
   , total_cells(0)
  {}

  CUDA Tables(AType uid, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : Tables(uid, sub->aty(), sub, alloc)
  {}

  template<class A2, class U2, class Alloc2, class... Allocators>
  CUDA NI Tables(const Tables<A2, U2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , store_aty(other.store_aty)
   , sub(deps.template clone<sub_type>(other.sub))
   , headers(other.headers, deps.template get_allocator<allocator_type>())
   , tell_tables(other.tell_tables, deps.template get_allocator<allocator_type>())
   , ask_tables(other.ask_tables, deps.template get_allocator<allocator_type>())
   , eliminated_rows(other.eliminated_rows, deps.template get_allocator<allocator_type>())
   , table_idx_to_column(other.table_idx_to_column, deps.template get_allocator<allocator_type>())
   , column_to_table_idx(other.column_to_table_idx, deps.template get_allocator<allocator_type>())
   , total_cells(other.total_cells)
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return headers.get_allocator();
  }

  CUDA sub_ptr subdomain() const {
    return sub;
  }

  CUDA local::BDec is_bot() const {
    return tell_tables.size() == 0 && sub->is_bot();
  }

  CUDA local::BInc is_top() const {
    for(int i = 0; i < eliminated_rows.size(); ++i) {
      if(eliminated_rows[i].count() == tell_tables[i].size()) {
        return true;
      }
    }
    return sub->is_top();
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return Tables{atype, battery::allocate_shared<sub_type>(alloc, sub_type::bot(atype_sub, sub_alloc)), alloc};
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return Tables{atype, battery::allocate_shared<sub_type>(sub_alloc, sub_type::top(atype_sub, sub_alloc)), alloc};
  }

  template <class Env>
  CUDA static this_type bot(Env& env,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    AType atype_sub = env.extends_abstract_dom();
    AType atype = env.extends_abstract_dom();
    return bot(atype, atype_sub, alloc, sub_alloc);
  }

  template <class Env>
  CUDA static this_type top(Env& env,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    AType atype_sub = env.extends_abstract_dom();
    AType atype = env.extends_abstract_dom();
    return top(atype, atype_sub, alloc, sub_alloc);
  }

  template <class Alloc2>
  struct snapshot_type {
    using sub_snap_type = sub_type::template snapshot_type<Alloc2>;
    sub_snap_type sub_snap;
    size_t num_tables;
    size_t total_cells;

    snapshot_type(const snapshot_type<Alloc2>&) = default;
    snapshot_type(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(const snapshot_type<Alloc2>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc2& alloc = Alloc2())
      : sub_snap(other.sub_snap, alloc)
      , num_tables(other.num_tables)
    {}

    CUDA snapshot_type(sub_snap_type&& sub_snap, size_t num_tables, size_t total_cells)
      : sub_snap(std::move(sub_snap))
      , num_tables(num_tables)
      , total_cells(total_cells)
    {}
  };

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>(sub->snapshot(alloc), headers.size(), total_cells);
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    sub->restore(snap.sub_snap);
    total_cells = snap.total_cells;
    table_idx_to_column.resize(snap.num_tables + 1);
    headers.resize(snap.num_tables);
    column_to_table_idx.resize(table_idx_to_column.back());
    tell_tables.resize(snap.num_tables);
    ask_tables.resize(snap.num_tables);
    eliminated_rows.resize(snap.num_tables);
    for(int i = 0; i < eliminated_rows.size(); ++i) {
      eliminated_rows[i].reset();
    }
  }

  template <class F>
  CUDA void flatten_and(const F& f, typename F::Sequence& conjuncts) const {
    if(f.is(F::Seq) && f.sig() == AND) {
      for(int i = 0; i < f.seq().size(); ++i) {
        flatten_and(f.seq(i), conjuncts);
      }
    }
    else {
      conjuncts.push_back(f);
    }
  }

  template <class F>
  CUDA void flatten_or(const F& f, typename F::Sequence& disjuncts) const {
    if(f.is(F::Seq) && f.sig() == OR) {
      for(int i = 0; i < f.seq().size(); ++i) {
        flatten_or(f.seq(i), disjuncts);
      }
    }
    else {
      typename F::Sequence conjuncts{disjuncts.get_allocator()};
      flatten_and(f, conjuncts);
      if(conjuncts.size() > 1) {
        disjuncts.push_back(F::make_nary(AND, std::move(conjuncts)));
      }
      else {
        disjuncts.push_back(std::move(conjuncts[0]));
      }
    }
  }

  template <class F>
  CUDA F flatten(const F& f, const typename F::allocator_type& alloc) const {
    typename F::Sequence disjuncts{alloc};
    flatten_or(f, disjuncts);
    if(disjuncts.size() > 1) {
      return F::make_nary(OR, std::move(disjuncts));
    }
    else {
      return std::move(disjuncts[0]);
    }
  }

  template <IKind kind, bool diagnose = false, class F, class Env, class Alloc>
  CUDA NI bool interpret_atom(
    battery::vector<AVar, Alloc>& header,
    battery::vector<battery::vector<local_universe, Alloc>, Alloc>& tell_table,
    battery::vector<battery::vector<local_universe, Alloc>, Alloc>& ask_table,
    const F& f, Env& env, IDiagnostics& diagnostics) const
  {
    if(num_vars(f) != 1) {
      RETURN_INTERPRETATION_ERROR("Only unary formulas are supported in the cell of the table.");
    }
    else {
      auto x_opt = var_in(f, env);
      if(!x_opt.has_value() || !x_opt.value().avar_of(store_aty).has_value()) {
        RETURN_INTERPRETATION_ERROR("Undeclared variable.");
      }
      AVar x = x_opt.value().avar_of(store_aty).value();
      int idx = 0;
      for(; idx < header.size() && header[idx] != x; ++idx) {}
      // If it's a new variable not present in the previous rows, we add it in each row with bottom value.
      if(idx == header.size()) {
        header.push_back(x);
        for(int i = 0; i < tell_table.size(); ++i) {
          if constexpr(kind == IKind::TELL) {
            tell_table[i].push_back(local_universe::bot());
          }
          ask_table[i].push_back(local_universe::bot());
        }
      }
      local_universe ask_u{local_universe::bot()};
      if(ginterpret_in<IKind::ASK, diagnose>(f, env, ask_u, diagnostics)) {
        ask_table.back()[idx].tell(ask_u);
        if constexpr(kind == IKind::TELL) {
          local_universe tell_u{local_universe::bot()};
          if(ginterpret_in<IKind::TELL, diagnose>(f, env, tell_u, diagnostics)) {
            tell_table.back()[idx].tell(tell_u);
          }
          else {
            return false;
          }
        }
      }
      else {
        return false;
      }
    }
    return true;
  }

public:
  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f2, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    F f = flatten(f2, env.get_allocator());
    using Alloc = typename I::allocator_type;
    if(f.is(F::Seq) && f.sig() == OR) {
      battery::vector<AVar, Alloc> header(intermediate.get_allocator());
      battery::vector<battery::vector<local_universe, Alloc>, Alloc> tell_table(intermediate.get_allocator());
      battery::vector<battery::vector<local_universe, Alloc>, Alloc> ask_table(intermediate.get_allocator());
      for(int i = 0; i < f.seq().size(); ++i) {
        // Add a row in the table.
        tell_table.push_back(battery::vector<local_universe, Alloc>(header.size(), local_universe::bot(), intermediate.get_allocator()));
        ask_table.push_back(battery::vector<local_universe, Alloc>(header.size(), local_universe::bot(), intermediate.get_allocator()));
        if(f.seq(i).is(F::Seq) && f.seq(i).sig() == AND) {
          const auto& row = f.seq(i).seq();
          for(int j = 0; j < row.size(); ++j) {
            size_t error_ctx = diagnostics.num_suberrors();
            if(!interpret_atom<kind, diagnose>(header, tell_table, ask_table, row[j], env, diagnostics)) {
              if(!sub->template interpret<kind, diagnose>(f2, env, intermediate.sub, diagnostics)) {
                return false;
              }
              diagnostics.cut(error_ctx);
              return true;
            }
          }
        }
        else {
          size_t error_ctx = diagnostics.num_suberrors();
          if(!interpret_atom<kind, diagnose>(header, tell_table, ask_table, f.seq(i), env, diagnostics)) {
            if(!sub->template interpret<kind, diagnose>(f2, env, intermediate.sub, diagnostics)) {
              return false;
            }
            diagnostics.cut(error_ctx);
            return true;
          }
        }
      }
      intermediate.headers.push_back(std::move(header));
      if constexpr(kind == IKind::TELL) {
        intermediate.tell_tables.push_back(std::move(tell_table));
      }
      intermediate.ask_tables.push_back(std::move(ask_table));
      return true;
    }
    else {
      return sub->template interpret<kind, diagnose>(f, env, intermediate.sub, diagnostics);
    }
  }

  template <bool diagnose = false, class F, class Env, class Alloc>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc>& ask, IDiagnostics& diagnostics) const {
    return interpret<IKind::ASK, diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc>& tell, IDiagnostics& diagnostics) const {
    return interpret<IKind::TELL, diagnose>(f, env, tell, diagnostics);
  }

  CUDA const sub_universe_type& operator[](int x) const {
    return (*sub)[x];
  }

  CUDA size_t vars() const {
    return sub->vars();
  }

private:
  template <IKind kind>
  CUDA sub_local_universe convert(const local_universe& x) const {
    if constexpr(std::is_same_v<universe_type, sub_universe_type>) {
      return x;
    }
    else {
      VarEnv<battery::standard_allocator> env;
      IDiagnostics diagnostics;
      sub_local_universe v{sub_local_universe::bot()};
      bool succeed = ginterpret_in<kind>(x.deinterpret(AVar{}, env), env, v, diagnostics);
      assert(succeed);
      return v;
    }
  }

public:
  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    if(t.headers.size() > 0) {
      has_changed.tell_top();
    }
    sub->tell(t.sub, has_changed);
    for(int i = 0; i < t.headers.size(); ++i) {
      headers.push_back(battery::vector<AVar, allocator_type>(t.headers[i], get_allocator()));
      for(int j = 0; j < headers[i].size(); ++j) {
        column_to_table_idx.push_back(i);
      }
      table_idx_to_column.push_back(table_idx_to_column.back() + t.tell_tables[i][0].size());
      tell_tables.push_back(table_type(t.tell_tables[i], get_allocator()));
      ask_tables.push_back(table_type(t.ask_tables[i], get_allocator()));
      eliminated_rows.push_back(bitset_type(tell_tables.back().size(), get_allocator()));
      total_cells += tell_tables.back().size() * tell_tables.back()[0].size();
    }
    return *this;
  }

  template <class Alloc>
  CUDA this_type& tell(const tell_type<Alloc>& t)  {
    local::BInc has_changed;
    return tell(t, has_changed);
  }

  CUDA this_type& tell(AVar x, const sub_universe_type& dom) {
    sub->tell(x, dom);
    return *this;
  }

  template <class Mem>
  CUDA this_type& tell(AVar x, const sub_universe_type& dom, BInc<Mem>& has_changed) {
    sub->tell(x, dom, has_changed);
    return *this;
  }

private:
  template <class Alloc>
  CUDA local::BInc ask(const battery::vector<battery::vector<AVar, Alloc>, Alloc>& header,
   const battery::vector<battery::vector<battery::vector<universe_type, Alloc>, Alloc>, Alloc>& ask_tables) const
  {
    for(int i = 0; i < ask_tables.size(); ++i) {
      bool table_entailed = false;
      for(int j = 0; j < ask_tables[i].size() && !table_entailed; ++j) {
        bool row_entailed = true;
        for(int k = 0; k < ask_tables[i][j].size(); ++k) {
          if(!(sub->project(headers[i][k]) >= convert<IKind::ASK>(ask_tables[i][j][k]))) {
            row_entailed = false;
            break;
          }
        }
        if(row_entailed) {
          table_entailed = true;
        }
      }
      if(!table_entailed) {
        return false;
      }
    }
    return true;
  }

public:
  template <class Alloc>
  CUDA local::BInc ask(const ask_type<Alloc>& a) const {
    return ask(a.headers, a.ask_tables) && sub->ask(a.sub);
  }

  template <class Mem>
  CUDA void crefine(size_t table_num, size_t col, BInc<Mem>& has_changed) {
    sub_local_universe u{sub_local_universe::top()};
    for(int j = 0; j < tell_tables[table_num].size(); ++j) {
      if(!eliminated_rows[table_num].test(j)) {
        u.dtell(convert<IKind::TELL>(tell_tables[table_num][j][col]));
      }
    }
    sub->tell(headers[table_num][col], u, has_changed);
  }

  template <class Mem>
  CUDA void lrefine(size_t table_num, size_t row, size_t col, BInc<Mem>& has_changed)
  {
    if(!eliminated_rows[table_num].test(row))
    {
      if(join(convert<IKind::ASK>(ask_tables[table_num][row][col]), sub->project(headers[table_num][col])).is_top()) {
        eliminated_rows[table_num].set(row);
        has_changed.tell_top();
      }
    }
  }

  CUDA size_t num_refinements() const {
    return
      sub->num_refinements() +
      column_to_table_idx.size() + // number of crefine (one per column).
      total_cells; // number of lrefine (one per cell).
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    assert(i < num_refinements());
    if(i < sub->num_refinements()) {
      sub->refine(i, has_changed);
    }
    else {
      i -= sub->num_refinements();
      if(i < column_to_table_idx.size()) {
        crefine(column_to_table_idx[i], i - table_idx_to_column[column_to_table_idx[i]], has_changed);
      }
      else {
        i -= column_to_table_idx.size();
        size_t table_num = 0;
        bool unfinished = true;
        // This loop computes the table number of the cell `i`, we avoid stopping the loop earlier to avoid thread divergence.
        for(int j = 0; j < tell_tables.size(); ++j) {
          size_t dim_table = tell_tables[j].size() * tell_tables[j][0].size();
          unfinished &= (i >= dim_table);
          i -= (unfinished ? dim_table : 0);
          table_num += (unfinished ? 1 : 0);
        }
        lrefine(table_num, i / tell_tables[table_num][0].size(), i % tell_tables[table_num][0].size(), has_changed);
      }
    }
  }

  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    // Check all remaining row are entailed.
    return ask(headers, ask_tables) && sub->is_extractable(strategy);
  }

  /** Extract an under-approximation if the last node popped \f$ a \f$ is an under-approximation.
   * If `B` is a search tree, the under-approximation consists in a search tree \f$ \{a\} \f$ with a single node, in that case, `ua` must be different from `top`. */
  template <class B>
  CUDA void extract(B& ua) const {
    if constexpr(impl::is_table_like<B>::value) {
      sub->extract(*ua.sub);
    }
    else {
      sub->extract(ua);
    }
  }

  CUDA sub_universe_type project(AVar x) const {
    return sub->project(x);
  }

  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    F sub_f = sub->deinterpret(env);
    typename F::Sequence seq{env.get_allocator()};
    if(sub_f.is(F::Seq) && sub_f.sig() == AND) {
      seq = std::move(sub_f.seq());
    }
    else {
      seq.push_back(std::move(sub_f));
    }
    for(int i = 0; i < headers.size(); ++i) {
      typename F::Sequence disjuncts{env.get_allocator()};
      for(int j = 0; j < tell_tables[i].size(); ++j) {
        if(!eliminated_rows[i].test(j)) {
          typename F::Sequence conjuncts{env.get_allocator()};
          for(int k = 0; k < tell_tables[i][j].size(); ++k) {
            if(!(sub->project(headers[i][k]) >= convert<IKind::ASK>(ask_tables[i][j][k]))) {
              conjuncts.push_back(tell_tables[i][j][k].deinterpret(headers[i][k], env));
            }
          }
          disjuncts.push_back(F::make_nary(AND, std::move(conjuncts), aty()));
        }
      }
      seq.push_back(F::make_nary(OR, std::move(disjuncts), aty()));
    }
    return F::make_nary(AND, std::move(seq));
  }
};

}

#endif
