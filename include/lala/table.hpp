// Copyright 2022 Pierre Talbot

#ifndef LALA_POWER_SEARCH_TREE_HPP
#define LALA_POWER_SEARCH_TREE_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"

namespace lala {

template <class A, class U, class Alloc> class Talbe;
namespace impl {
  template <class>
  struct is_table_like {
    static constexpr bool value = false;
  };
  template<class A, class U, class Alloc>
  struct is_table_like<Table<A, U, Alloc>> {
    static constexpr bool value = true;
  };
}

/** The table abstract domain is designed to represent predicates in extension by listing all their solutions explicitly.
 * It is inspired by the table global constraint and generalizes it by lifting each element of the table to a lattice element.
 * We expect `U` to be equally or less expressive than `A::universe_type`, this is because we compute the meet in `A::universe_type` and not in `U`.
 */
template <class A, class U = typename A::universe_type, class Allocator = typename A::allocator_type>
class Table {
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
  using this_type = Table<sub_type, universe_type, allocator_type>;

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
  constexpr static const char* name = "Table";

  using table_type = battery::vector<
    battery::vector<universe_type, allocator_type>,
    allocator_type>;
  using table_collection_type = battery::vector<table_type, allocator_type>;
  using bitset_type = battery::dynamic_bitset<memory_type, allocator_type>;

private:
  AType atype;
  sub_ptr sub;

  battery::vector<battery::vector<AVar, allocator_type>, allocator_type> headers;
  table_collection_type tell_tables;
  table_collection_type ask_tables;
  battery::vector<bitset_type, allocator_type> eliminated_rows;
  // See `refine`.
  battery::vector<size_t, allocator_type> table_to_lrefine;
  battery::vector<size_t, allocator_type> crefine_to_table;

public:
  template <class Alloc>
  struct tell_type {
    using allocator_type = Alloc;

    typename A::template tell_type<Alloc> sub_tell;

    battery::vector<battery::vector<AVar, Alloc>, Alloc> headers;

    battery::vector<battery::vector<
      battery::vector<universe_type, Alloc>,
    Alloc>, Alloc> tell_tables;

    battery::vector<battery::vector<
      battery::vector<universe_type, Alloc>,
    Alloc>, Alloc> ask_tables;

    CUDA tell_type(const Alloc& alloc = Alloc{}): sub_tell(alloc), split_tell(alloc) {}
    tell_type(const tell_type&) = default;
    tell_type(tell_type&&) = default;
    tell_type& operator=(tell_type&&) = default;
    tell_type& operator=(const tell_type&) = default;

    template <class TableTellType>
    CUDA NI tell_type(const TableTellType& other, const Alloc& alloc = Alloc{})
      : sub_tell(other.sub_tell, alloc)
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

  template<class Alloc>
  using ask_type = typename A::template ask_type<Alloc>;

  template <class A2, class U2, class Alloc2>
  friend class Table;

public:
  CUDA Table(AType uid, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : atype(uid)
   , sub(std::move(sub))
   , headers(alloc)
   , tell_tables(alloc)
   , ask_tables(alloc)
   , eliminated_rows(alloc)
   , table_to_lrefine({0}, alloc)
   , crefine_to_table(alloc)
  {}

  template<class A2, class U2, class Alloc2, class... Allocators>
  CUDA NI Table(const Table<A2, U2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , sub(deps.template clone<sub_type>(other.sub))
   , headers(other.headers, deps.template get_allocator<allocator_type>())
   , tell_tables(other.tell_tables, deps.template get_allocator<allocator_type>())
   , ask_tables(other.ask_tables, deps.template get_allocator<allocator_type>())
   , eliminated_rows(other.eliminated_rows, deps.template get_allocator<allocator_type>())
   , table_to_lrefine(other.table_to_lrefine, deps.template get_allocator<allocator_type>())
   , crefine_to_table(other.crefine_to_table, deps.template get_allocator<allocator_type>())
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return headers.get_allocator();
  }

  CUDA local::BDec is_bot() const {
    return sub->is_bot();
  }

  CUDA local::BInc is_top() const {
    return sub->is_top();
  }

  template <class Alloc2>
  struct snapshot_type {
    using sub_snap_type = sub_type::template snapshot_type<Alloc2>;
    sub_snap_type sub_snap;
    size_t num_tables;

    snapshot_type(const snapshot_type<Alloc2>&) = default;
    snapshot_type(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(const snapshot_type<Alloc2>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc2& alloc = Alloc2())
      : sub_snap(other.sub_snap, alloc)
      , num_tables(other.num_tables)
    {}

    CUDA snapshot_type(sub_snap_type&& sub_snap, size_t num_tables)
      : sub_snap(std::move(sub_snap))
      , num_tables(num_tables)
    {}
  };

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>(sub->snapshot(alloc), headers.size());
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    sub->restore(snap.sub_snap);
    table_to_lrefine.resize(snap.num_tables + 1);
    headers.resize(snap.num_tables);
    size_t total_columns = 0;
    for(int i = 0; i < headers.size(); ++i) {
      total_columns += headers[i].size();
    }
    crefine_to_table.resize(total_columns);
    tell_tables.resize(snap.num_tables);
    ask_tables.resize(snap.num_tables);
    eliminated_rows.resize(snap.num_tables);
  }

public:
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    if(f.is(F::Seq) && f.sig() == OR) {
      battery::vector<AVar, Alloc2> header(tell.get_allocator());
      battery::vector<battery::vector<local_universe, Alloc>, Alloc> tell_table(tell.get_allocator());
      battery::vector<battery::vector<local_universe, Alloc>, Alloc> ask_table(tell.get_allocator());
      for(int i = 0; i < f.seq().size(); ++i) {
        if(f.seq(i).is(F::Seq) && f.seq(i).sig() == AND) {
          // Add a row in the table.
          tell_table.push_back(battery::vector<local_universe, Alloc>(header.size(), local_universe::bot(), tell.get_allocator()));
          ask_table.push_back(battery::vector<local_universe, Alloc>(header.size(), local_universe::bot(), tell.get_allocator()));
          const auto& row = f.seq(i).seq();
          for(int j = 0; j < row.size(); ++j) {
            if(num_vars(row[j]) != 1) {
              RETURN_INTERPRETATION_ERROR("Only unary formulas are supported in the cell of the table.");
            }
            else {
              // TODO: return an error if the variable is not declared in env.
              AVar x = var_in(row[j], env).value();
              int idx = 0;
              for(; idx < header.size() && header[idx] != x; ++idx) {}
              // If it's a new variable not present in the previous rows, we add it in each row with bottom value.
              if(idx == header.size()) {
                header.push_back(x);
                for(int i = 0; i < tell_table.size(); ++i) {
                  tell_table[i].push_back(local_universe::bot());
                  ask_table[i].push_back(local_universe::bot());
                }
              }
              local_universe tell_u{local::universe::bot()};
              local_universe ask_u{local::universe::bot()};
              if( local_universe::interpret_tell<diagnose>(row[j], env, tell_u, diagnostics)
               && local_universe::interpret_ask<diagnose>(row[j], env, ask_u, diagnostics))
              {
                tell_table[i][j].tell(tell_u);
                ask_table[i][j].tell(ask_u);
              }
              else {
                return false;
              }
            }
          }
        }
        else {
          // To improve, we could support it by interpreting a single atom.
          RETURN_INTERPRETATION_ERROR("Only disjunction of conjunctions are supported.");
        }
      }
      tell.headers.push_back(std::move(header));
      tell.tell_tables.push_back(std::move(tell_table));
      tell.ask_tables.push_back(std::move(ask_table));
      return true;
    }
    else {
      return sub->template interpret_tell<diagnose>(f, env, tell.sub_tell, diagnostics);
    }
  }

  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc2>& ask, IDiagnostics& diagnostics) const {
    return interpret_tell<diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
  }

  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    if constexpr(kind == IKind::TELL) {
      return interpret_tell<diagnose>(f, env, intermediate, diagnostics);
    }
    else {
      return interpret_ask<diagnose>(f, env, intermediate, diagnostics);
    }
  }

public:
  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    if(t.headers.size() > 0) {
      has_changed.tell_top();
    }
    sub->tell(t.sub_tell, has_changed);
    for(int i = 0; i < t.headers.size(); ++i) {
      headers.push_back(t.headers[i]);
      for(int j = 0; j < headers[i].size(); ++j) {
        crefine_to_table.push_back(i);
      }
      table_to_lrefine.push_back(table_to_lrefine.back() + t.tell_tables[i].size() * t.tell_tables[i][0].size());
      tell_tables.push_back(t.tell_tables[i]);
      ask_tables.push_back(t.ask_tables[i]);
      eliminated_rows.push_back(bitset_type(tell_tables.back()[0].size(), get_allocator()));
    }
    return *this;
  }

  template <class Mem>
  CUDA void crefine(size_t table_num, size_t col, BInc<Mem>& has_changed) {
    sub_local_universe u{sub_local_universe::top()};
    for(int j = 0; j < tell_tables[table_num].size(); ++j) {
      if(!eliminated_rows[table_num].test(j)) {
        if constexpr(std::is_same_v<universe_type, sub_universe_type>) {
          u.dtell(tell_tables[table_num][j][col]);
        }
        else {
          VarEnv<standard_allocator> env;
          auto f = tell_tables[table_num][j][col].deinterpret(headers[table_num][col], env);
          IDiagnostics diagnostics;
          sub_local_universe v{sub_local_universe::bot()};
          if(sub_local_universe::interpret_tell(f, env, v, diagnostics)) {
            u.dtell(v);
          }
        }
      }
    }
    sub->tell(headers[table_num][col], u, has_changed);
  }

  template <class Mem>
  CUDA void lrefine(size_t table_num, size_t col, size_t row, BInc<Mem>& has_changed)
  {
    if(!eliminated_rows[table_num].test(row))
    {
      if constexpr(std::is_same_v<universe_type, sub_universe_type>) {
        if(join(ask_tables[table_num][row][col], sub->project(headers[table_num][col])).is_top()) {
          eliminated_rows[table_num].set(row)
          has_changed.tell_top();
        }
      }
      else {
        VarEnv<standard_allocator> env;
        auto f = ask_tables[table_num][j][col].deinterpret(headers[table_num][col], env);
        IDiagnostics diagnostics;
        sub_local_universe x{sub->project(headers[table_num][col])};
        sub_local_universe::interpret_ask(f, env, x, diagnostics);
        if(x.is_top()) {
          eliminated_rows[table_num].set(row)
          has_changed.tell_top();
        }
      }
    }
  }

  CUDA size_t num_refinements() const {
    return
      crefine_to_table.size() + // number of crefine (one per column).
      table_to_lrefine.back(); // number of lrefine (one per cell).
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    if(i < crefine_to_table.size()) {
      crefine(crefine_to_table[i], i % crefine_to_table[i], has_changed);
    }
    else {
      size_t table_num = 0;
      for(; table_num < table_to_lrefine.size() && table_to_lrefine[table_num] <= i; ++table_num) {}
      lrefine(table_num, (i - table_to_lrefine[table_num - 1]) / tell_tables[table_num - 1][0].size(), (i - table_to_lrefine[table_num - 1]) % tell_tables[table_num - 1][0].size(), has_changed);
    }
  }

  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    // Check all remaining row are entailed.
    return sub->is_extractable(strategy);
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

  CUDA universe_type project(AVar x) const {
    return sub->project(x);
  }

  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    F sub_f = sub->deinterpret(env);
    typename F::Sequence seq{env.get_allocator()};
    if(sub_f.is(F::Seq) && sub_f.sig() == AND) {
      for(int i = 0; i < sub_f.seq().size(); ++i) {
        seq.push_back(sub_f.seq(i));
      }
    }
    else {
      seq.push_back(sub_f);
    }
    for(int i = 0; i < headers.size(); ++i) {
      // TODO
    }
    return F::make_nary(AND, std::move(seq), aty());
  }
};

}

#endif
