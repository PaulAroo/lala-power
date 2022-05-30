// Copyright 2022 Pierre Talbot

#ifndef SEARCH_TREE_HPP
#define SEARCH_TREE_HPP

#include "ast.hpp"
#include "z.hpp"
#include "arithmetic.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

namespace lala {

template <class A, class Split>
class SearchTree {
public:
  using Allocator = typename A::Allocator;
  using BranchType = typename Split::BranchType;
  using TellType = typename A::TellType;
  using Universe = typename A::Universe;
  using APtr = battery::shared_ptr<A, Allocator>;
  using SplitPtr = battery::shared_ptr<Split, Allocator>;
  using this_type = SearchTree<A, Split>;
  using Env = typename A::Env;

private:
  AType uid_;
  // `a` reflects the current node of the search tree being refined and expanded.
  // If the search tree is `top` (i.e., empty), then `a` is equal to `nullptr`.
  APtr a;
  SplitPtr split;
  battery::vector<BranchType, Allocator> stack;
  A::Snapshot root;
  // Formulas to be added to root on backtracking.
  battery::vector<TellType, Allocator> root_formulas;

public:
  CUDA SearchTree(AType uid, APtr a, SplitPtr split)
   : uid_(uid), a(std::move(a)), split(std::move(split)),
     stack(this->a->get_allocator()), root_formulas(this->a->get_allocator())
  {}

  template<class Alloc2>
  CUDA SearchTree(const SearchTree<A, Alloc2>& other, AbstractDeps<Allocator>& deps)
   : uid_(other.uid_), a(deps.clone(other.a)), split(deps.clone(other.split)),
     stack(other.stack), root(other.root), root_formulas(other.root_formulas)
  {}

  CUDA AType uid() const {
    return uid_;
  }

  CUDA Allocator get_allocator() const {
    return a->get_allocator();
  }

  CUDA BDec is_singleton() const {
    return stack.empty() && bool(a);
  }

  CUDA BDec is_bot() const {
    // We need short-circuit using && (instead of `land`) due to `a` possibly a null pointer.
    return is_singleton().value() && a->is_bot().value();
  }

  CUDA BInc is_top() const {
    return !bool(a);
  }

  template <class F>
  CUDA thrust::optional<TellType> interpret(const F& f) {
    if(is_top().guard()) {
      return {};  // We could actually interpret `f` as `false` instead.
    }
    auto r = a->interpret(f);
    if(r.has_value()) {
      split->interpret();
    }
    return std::move(r);
  }

  CUDA this_type& tell(TellType&& t, BInc& has_changed) {
    if(!is_top().guard()) {
      if(lnot(is_singleton()).guard()) {
        // We will add `t` to root when we backtrack (see pop) and have a chance to modify the root node.
        root_formulas.push_back(t);
      }
      // Nevertheless, the rest of the subtree to be explored is still updated with `t`.
      a->tell(std::move(t), has_changed);
      has_changed.tell(BInc::top());
    }
    return *this;
  }

  /** The refinement of `a` and `split` is not done here, and if needed, should be done before calling this method.
   * This refinement operator performs one iteration of \f$ \mathit{pop} \circ \mathit{push} \circ \mathit{split} \f$.
   * In short, it initializes `a` to the next node of the search tree.
   * If we observe `a` from the outside of this domain, `a` can backtrack, and therefore does not always evolve extensively and monotonically.
   * Nevertheless, the refinement operator of the search tree abstract domain is extensive and monotonic (if split is) over the search tree. */
  CUDA void refine(BInc& has_changed) {
    pop(push(split->split()), has_changed);
  }

  /** Extract an under-approximation if the last node popped \f$ a \f$ is an under-approximation.
   * The under-approximation consists in a search tree \f$ \{a\} \f$ with a single node.
   * \pre `ua` must be different from `top`. */
  CUDA bool extract(this_type& ua) const {
    if(!is_top().guard()) {
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
  CUDA Universe project(AVar x) const {
    if(is_top().guard()) {
      return Universe::top();
    }
    else {
      if(is_singleton().value()) {
        return a->project(x);
      }
      else {
        assert(false);
        return Universe::bot();
        /** The problem with the method below is that we need to modify `a`, so project is not const anymore.
         * That might be problematic to modify `a` for a projection if it is currently being refined...
         * Perhaps need to copy `a` (inefficient), or request a projection in the snapshot directly. */
        // a->restore(root);
        // Universe u = a->project(x);
        // BInc has_changed = BInc::bot();
        // replay(has_changed);
        // return u;
      }
    }
  }

  CUDA const Env& environment() const {
    return a->environment();
  }

private:
  /** \return `true` if the current node is pruned, and `false` if a new branch was pushed. */
  CUDA bool push(BranchType&& branch) {
    if(branch.size() > 0) {
      if(is_singleton().value()) {
        root = a->snapshot();
      }
      stack.push_back(std::move(branch));
      return false;
    }
    return true;
  }

  /** If the current node was pruned, we need to backtrack, otherwise we just consider the next node along the branch. */
  CUDA void pop(bool pruned, BInc& has_changed) {
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
  CUDA void commit_left(BInc& has_changed) {
    assert(bool(a));
    a->tell(TellType(stack.back().next()), has_changed);
  }

  /** We explore the next node of the search tree available (after we backtracked, so it cannot be a left node). */
  CUDA void commit_right(BInc& has_changed) {
    if(!stack.empty()) {
      assert(bool(a));
      stack.back().next();
      replay(has_changed);
    }
  }

  /** Goes from the current node to root. */
  CUDA void backtrack(BInc& has_changed) {
    while(!stack.empty() && !stack.back().has_next()) {
      stack.pop_back();
    }
    if(!stack.empty()) {
      a->restore(root);
      tell_root(has_changed);
    }
    else if(a) {
      a = nullptr;
      has_changed.tell(BInc::top());
    }
  }

  /** We do not always have access to the root node, so formulas that are added to the search tree are kept in `root_formulas`.
   * During backtracking, root is available through `a`, and we add to root the formulas stored until now, so they become automatically available to the remaining nodes in the search tree. */
  CUDA void tell_root(BInc& has_changed) {
    if(root_formulas.size() > 0) {
      for(int i = 0; i < root_formulas.size(); ++i) {
        a->tell(std::move(root_formulas[i]), has_changed);
      }
      root_formulas.clear();
      root = a->snapshot();
    }
  }

  /** Goes from `root` to the new node to be explored. */
  CUDA void replay(BInc& has_changed) {
    for(int i = 0; i < stack.size(); ++i) {
      assert(i < stack.size());
      a->tell(TellType(stack[i].current()), has_changed);
    }
  }
};

}

#endif
