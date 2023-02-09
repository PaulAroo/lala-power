// Copyright 2022 Pierre Talbot

#ifndef SEARCH_TREE_HPP
#define SEARCH_TREE_HPP

#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "copy_dag_helper.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

namespace lala {

template <class A, class Split>
class SearchTree {
public:
  using allocator_type = typename A::allocator_type;
  using branch_type = typename Split::branch_type;
  template <class Alloc>
  using tell_type = typename A::tell_type<Alloc>;
  using universe_type = typename A::universe_type;
  using sub_type = A;
  using sub_ptr = battery::shared_ptr<A, allocator_type>;
  using split_ptr = battery::shared_ptr<Split, allocator_type>;
  using this_type = SearchTree<A, Split>;

  template<class F, class Env>
  using iresult = typename A::iresult<F, Env>;
  constexpr static const char* name = "SearchTree";

private:
  AType atype;
  // `a` reflects the current node of the search tree being refined and expanded.
  // If the search tree is `top` (i.e., empty), then `a` is equal to `nullptr`.
  sub_ptr a;
  split_ptr split;
  battery::vector<branch_type, allocator_type> stack;
  typename A::snapshot_type<allocator_type> root;
  // Formulas to be added to root on backtracking.
  battery::vector<tell_type<allocator_type>, allocator_type> root_formulas;

public:
  CUDA SearchTree(AType uid, sub_ptr a, split_ptr split)
   : atype(uid), a(std::move(a)), split(std::move(split)),
     stack(this->a->get_allocator()), root_formulas(this->a->get_allocator())
  {}

  template<class Alloc2>
  CUDA SearchTree(const SearchTree<A, Alloc2>& other, AbstractDeps<allocator_type>& deps)
   : atype(other.atype), a(deps.clone(other.a)), split(deps.clone(other.split)),
     stack(other.stack), root(other.root), root_formulas(other.root_formulas)
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return a->get_allocator();
  }

  CUDA local::BDec is_singleton() const {
    return stack.empty() && bool(a);
  }

  CUDA local::BDec is_bot() const {
    // We need short-circuit using && (instead of `land`) due to `a` possibly a null pointer.
    return is_singleton() && a->is_bot();
  }

  CUDA local::BInc is_top() const {
    return !bool(a);
  }

  template <class F, class Env>
  CUDA iresult<F, Env> interpret_in(const F& f, Env& env) {
    if(is_top()) {
      return iresult<F, Env>(IError<F>(true, name, "The current abstract element is `top`.", f));
    }
    auto r = a->interpret_in(f, env);
    if(r.has_value()) {
      split->interpret_in(env);
    }
    return std::move(r);
  }

  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    if(!is_top()) {
      if(!is_singleton()) {
        // We will add `t` to root when we backtrack (see `pop`) and have a chance to modify the root node.
        root_formulas.push_back(t);
      }
      // Nevertheless, the rest of the subtree to be explored is still updated with `t`.
      a->tell(t, has_changed);
      has_changed.tell_top();
    }
    return *this;
  }

  /** The refinement of `a` and `split` is not done here, and if needed, should be done before calling this method.
   * This refinement operator performs one iteration of \f$ \mathit{pop} \circ \mathit{push} \circ \mathit{split} \f$.
   * In short, it initializes `a` to the next node of the search tree.
   * If we observe `a` from the outside of this domain, `a` can backtrack, and therefore does not always evolve extensively and monotonically.
   * Nevertheless, the refinement operator of the search tree abstract domain is extensive and monotonic (if split is) over the search tree. */
  template <class Env, class Mem>
  CUDA void refine(Env& env, BInc<Mem>& has_changed) {
    pop(push(split->split(env)), has_changed);
  }

  /** Extract an under-approximation if the last node popped \f$ a \f$ is an under-approximation.
   * The under-approximation consists in a search tree \f$ \{a\} \f$ with a single node.
   * \pre `ua` must be different from `top`. */
  CUDA bool extract(this_type& ua) const {
    if(!is_top()) {
      assert(bool(ua.a));
      if(a->extract(*ua.a)) {
        ua.stack.clear();
        ua.root_formulas.clear();
        return true;
      }
    }
    return false;
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
        root = a->template snapshot<allocator_type>();
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
      a->restore(root);
      tell_root(has_changed);
    }
    else if(a) {
      a = nullptr;
      has_changed.tell_top();
    }
  }

  /** We do not always have access to the root node, so formulas that are added to the search tree are kept in `root_formulas`.
   * During backtracking, root is available through `a`, and we add to root the formulas stored until now, so they become automatically available to the remaining nodes in the search tree. */
  template <class Mem>
  CUDA void tell_root(BInc<Mem>& has_changed) {
    if(root_formulas.size() > 0) {
      for(int i = 0; i < root_formulas.size(); ++i) {
        a->tell(root_formulas[i], has_changed);
      }
      root_formulas.clear();
      root = a->template snapshot<allocator_type>();
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
