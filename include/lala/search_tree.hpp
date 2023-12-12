// Copyright 2022 Pierre Talbot

#ifndef LALA_POWER_SEARCH_TREE_HPP
#define LALA_POWER_SEARCH_TREE_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"
#include "lala/vstore.hpp"

#include "split_strategy.hpp"

namespace lala {
template <class A, class S, class Allocator> class SearchTree;
namespace impl {
  template <class>
  struct is_search_tree_like {
    static constexpr bool value = false;
  };
  template<class A, class S, class Alloc>
  struct is_search_tree_like<SearchTree<A, S, Alloc>> {
    static constexpr bool value = true;
  };
}

template <class A, class Split, class Allocator = typename A::allocator_type>
class SearchTree {
public:
  using allocator_type = Allocator;
  using sub_allocator_type = typename A::allocator_type;
  using split_type = Split;
  using branch_type = typename split_type::branch_type;
  using universe_type = typename A::universe_type;
  using sub_type = A;
  using sub_ptr = abstract_ptr<sub_type>;
  using split_ptr = abstract_ptr<split_type>;
  using this_type = SearchTree<sub_type, split_type, allocator_type>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = sub_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  // The next properties should be checked more seriously, relying on the sub-domain might be uneccessarily restrictive.
  constexpr static const bool preserve_join = sub_type::preserve_join;
  constexpr static const bool preserve_meet = sub_type::preserve_meet;
  constexpr static const bool injective_concretization = sub_type::injective_concretization;
  constexpr static const bool preserve_concrete_covers = sub_type::preserve_concrete_covers;
  constexpr static const char* name = "SearchTree";

  template <class Alloc>
  struct tell_type {
    typename A::template tell_type<Alloc> sub_tell;
    typename split_type::template tell_type<Alloc> split_tell;
    CUDA tell_type(const Alloc& alloc = Alloc{}): sub_tell(alloc), split_tell(alloc) {}
    tell_type(const tell_type&) = default;
    tell_type(tell_type&&) = default;
    tell_type& operator=(tell_type&&) = default;
    tell_type& operator=(const tell_type&) = default;

    template <class SearchTreeTellType>
    CUDA NI tell_type(const SearchTreeTellType& other, const Alloc& alloc = Alloc{})
      : sub_tell(other.sub_tell, alloc), split_tell(other.split_tell, alloc)
    {}

    using allocator_type = Alloc;
    CUDA allocator_type get_allocator() const {
      return sub_tell.get_allocator();
    }

    template <class Alloc2>
    friend class tell_type;
  };

  template<class Alloc>
  using ask_type = typename A::template ask_type<Alloc>;


  template <class A2, class S2, class Alloc2>
  friend class SearchTree;

private:
  AType atype;
  // `a` reflects the current node of the search tree being refined and expanded.
  // If the search tree is `top` (i.e., empty), then `a` is equal to `nullptr`.
  sub_ptr a;
  split_ptr split;
  battery::vector<branch_type, allocator_type> stack;
  using sub_snapshot_type = sub_type::template snapshot_type<allocator_type>;
  using split_snapshot_type = split_type::template snapshot_type<allocator_type>;
  using root_type = battery::tuple<sub_snapshot_type, split_snapshot_type>;
  root_type root;

  // Tell formulas (and strategies) to be added to root on backtracking.
  struct root_tell_type {
    battery::vector<typename A::template tell_type<allocator_type>, allocator_type> sub_tells;
    battery::vector<typename split_type::template tell_type<allocator_type>, allocator_type> split_tells;
    CUDA root_tell_type(const allocator_type& alloc): sub_tells(alloc), split_tells(alloc) {}
    template <class RootTellType>
    CUDA root_tell_type(const RootTellType& other, const allocator_type& alloc)
     : sub_tells(other.sub_tells, alloc), split_tells(other.split_tells, alloc) {}
  };
  root_tell_type root_tell;

public:
  CUDA SearchTree(AType uid, sub_ptr a, split_ptr split, const allocator_type& alloc = allocator_type())
   : atype(uid)
   , a(std::move(a))
   , split(std::move(split))
   , stack(alloc)
   , root(battery::make_tuple(this->a->snapshot(alloc), this->split->snapshot(alloc)))
   , root_tell(alloc)
  {}

  template<class A2, class S2, class Alloc2, class... Allocators>
  CUDA NI SearchTree(const SearchTree<A2, S2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , a(deps.template clone<sub_type>(other.a))
   , split(deps.template clone<split_type>(other.split))
   , stack(other.stack, deps.template get_allocator<allocator_type>())
   , root(
      sub_snapshot_type(battery::get<0>(other.root), deps.template get_allocator<allocator_type>()),
      split_snapshot_type(battery::get<1>(other.root), deps.template get_allocator<allocator_type>()))
   , root_tell(other.root_tell, deps.template get_allocator<allocator_type>())
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return stack.get_allocator();
  }

  CUDA local::BDec is_singleton() const {
    return stack.empty() && bool(a);
  }

  CUDA local::BDec is_bot() const {
    // We need short-circuit using && due to `a` possibly a null pointer.
    return is_singleton() && a->is_bot();
  }

  CUDA local::BInc is_top() const {
    return !bool(a);
  }

  template <class Alloc2>
  struct snapshot_type {
    using sub_snap_type = sub_type::template snapshot_type<Alloc2>;
    using split_snap_type = split_type::template snapshot_type<Alloc2>;
    sub_snap_type sub_snap;
    split_snap_type split_snap;
    sub_ptr sub;

    snapshot_type(const snapshot_type<Alloc2>&) = default;
    snapshot_type(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(const snapshot_type<Alloc2>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc2& alloc = Alloc2())
      : sub_snap(other.sub_snap, alloc)
      , split_snap(other.split_snap, alloc)
      , sub(other.sub)
    {}

    CUDA snapshot_type(sub_snap_type&& sub_snap, split_snap_type&& split_snap, sub_ptr sub)
      : sub_snap(std::move(sub_snap))
      , split_snap(std::move(split_snap))
      , sub(sub)
    {}
  };

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    assert(is_singleton());
    return snapshot_type<Alloc2>(a->snapshot(alloc), split->snapshot(alloc), a);
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    a = snap.sub;
    a->restore(snap.sub_snap);
    split->restore(snap.split_snap);
    stack.clear();
    root = battery::make_tuple(
      a->snapshot(get_allocator()),
      split->snapshot(get_allocator()));
    root_tell = root_tell_type{get_allocator()};
  }

public:
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    assert(!is_top());
    if(f.is(F::ESeq) && f.esig() == "search") {
      return split->template interpret_tell<diagnose>(f, env, tell.split_tell, diagnostics);
    }
    else {
      return a->template interpret_tell<diagnose>(f, env, tell.sub_tell, diagnostics);
    }
  }

  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, Env& env, ask_type<Alloc2>& ask, IDiagnostics& diagnostics) const {
    assert(!is_top());
    return a->template interpret_ask<diagnose>(f, env, ask, diagnostics);
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

private:
  template <class Alloc, class Mem>
  CUDA void tell_current(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    a->tell(t.sub_tell, has_changed);
    split->tell(t.split_tell, has_changed);
  }

public:
  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    if(!is_top()) {
      if(!is_singleton()) {
        // We will add `t` to root when we backtrack (see `pop`) and have a chance to modify the root node.
        root_tell.sub_tells.push_back(t.sub_tell);
        root_tell.split_tells.push_back(t.split_tell);
      }
      // Nevertheless, the rest of the subtree to be explored is still updated with `t`.
      tell_current(t, has_changed);
    }
    return *this;
  }

  /** The refinement of `a` and `split` is not done here, and if needed, should be done before calling this method.
   * This refinement operator performs one iteration of \f$ \mathit{pop} \circ \mathit{push} \circ \mathit{split} \f$.
   * In short, it initializes `a` to the next node of the search tree.
   * If we observe `a` from the outside of this domain, `a` can backtrack, and therefore does not always evolve extensively and monotonically.
   * Nevertheless, the refinement operator of the search tree abstract domain is extensive and monotonic (if split is) over the search tree. */
  template <class Mem>
  CUDA void refine(BInc<Mem>& has_changed) {
    pop(push(split->split()), has_changed);
  }

  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    return !is_top() && a->is_extractable(strategy);
  }

  /** Extract an under-approximation if the last node popped \f$ a \f$ is an under-approximation.
   * If `B` is a search tree, the under-approximation consists in a search tree \f$ \{a\} \f$ with a single node, in that case, `ua` must be different from `top`. */
  template <class B>
  CUDA void extract(B& ua) const {
    if constexpr(impl::is_search_tree_like<B>::value) {
      assert(bool(ua.a));
      a->extract(*ua.a);
      ua.stack.clear();
      ua.root_tell.sub_tells.clear();
      ua.root_tell.split_tells.clear();
    }
    else {
      a->extract(ua);
    }
  }

  /** If the search tree is empty (\f$ \top \f$), we return \f$ \top_U \f$.
   * If the search tree consists of a single node \f$ \{a\} \f$, we return the projection of `x` in that node.
   * Projection in a search tree with multiple nodes is currently not supported (assert false). */
  CUDA universe_type project(AVar x) const {
    if(is_top()) {
      return universe_type::top();
    }
    else {
      if(is_singleton()) {
        return a->project(x);
      }
      else {
        assert(false);
        return universe_type::bot();
        /** The problem with the method below is that we need to modify `a`, so project is not const anymore.
         * That might be problematic to modify `a` for a projection if it is currently being refined...
         * Perhaps need to copy `a` (inefficient), or request a projection in the snapshot directly. */
        // a->restore(root);
        // universe_type u = a->project(x);
        // BInc has_changed = BInc::bot();
        // replay(has_changed);
        // return u;
      }
    }
  }

  /** \return the current depth of the search tree. The root node has a depth of 0. */
  CUDA size_t depth() const {
    return stack.size();
  }

private:
  /** \return `true` if the current node is pruned, and `false` if a new branch was pushed. */
  CUDA bool push(branch_type&& branch) {
    if(branch.size() > 0) {
      if(is_singleton()) {
        root = battery::make_tuple(
          a->snapshot(get_allocator()),
          split->snapshot(get_allocator()));
      }
      stack.push_back(std::move(branch));
      return false;
    }
    return true;
  }

  /** If the current node was pruned, we need to backtrack, otherwise we just consider the next node along the branch. */
  template <class Mem>
  CUDA void pop(bool pruned, BInc<Mem>& has_changed) {
    if(!pruned) {
      commit_left(has_changed);
    }
    else {
      backtrack(has_changed);
      commit_right(has_changed);
    }
  }

  /** Given the current node, commit to the node on the left.
   * If we are on the root node, we save a snapshot of root before committing to the left node. */
  template <class Mem>
  CUDA void commit_left(BInc<Mem>& has_changed) {
    assert(bool(a));
    a->tell(stack.back().next(), has_changed);
  }

  /** We explore the next node of the search tree available (after we backtracked, so it cannot be a left node). */
  template <class Mem>
  CUDA void commit_right(BInc<Mem>& has_changed) {
    if(!stack.empty()) {
      assert(bool(a));
      stack.back().next();
      replay(has_changed);
    }
  }

  /** Goes from the current node to root. */
  template <class Mem>
  CUDA void backtrack(BInc<Mem>& has_changed) {
    while(!stack.empty() && !stack.back().has_next()) {
      stack.pop_back();
    }
    if(!stack.empty()) {
      a->restore(battery::get<0>(root));
      split->restore(battery::get<1>(root));
      tell_root(has_changed);
    }
    else if(a) {
      a = nullptr;
      has_changed.tell_top();
    }
  }

  /** We do not always have access to the root node, so formulas that are added to the search tree are kept in `root_tell`.
   * During backtracking, root is available through `a`, and we add to root the formulas stored until now, so they become automatically available to the remaining nodes in the search tree. */
  template <class Mem>
  CUDA void tell_root(BInc<Mem>& has_changed) {
    if(root_tell.sub_tells.size() > 0 || root_tell.split_tells.size() > 0) {
      for(int i = 0; i < root_tell.sub_tells.size(); ++i) {
        a->tell(root_tell.sub_tells[i], has_changed);
      }
      for(int i = 0; i < root_tell.split_tells.size(); ++i) {
        split->tell(root_tell.split_tells[i], has_changed);
      }
      root_tell.sub_tells.clear();
      root_tell.split_tells.clear();
      // A new snapshot is necessary since we modified `a` and `split`.
      root = battery::make_tuple(
        a->snapshot(get_allocator()),
        split->snapshot(get_allocator()));
    }
  }

  /** Goes from `root` to the new node to be explored. */
  template <class Mem>
  CUDA void replay(BInc<Mem>& has_changed) {
    for(int i = 0; i < stack.size(); ++i) {
      a->tell(stack[i].current(), has_changed);
    }
  }
};

}

#endif
