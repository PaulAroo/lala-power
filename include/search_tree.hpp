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
  using TellType = A::TellType;

  using APtr = battery::shared_ptr<A, Allocator>;
  using SplitPtr = battery::shared_ptr<Split, Allocator>;
  using this_type = SearchTree<A, Split>;

private:
  AType uid_;
  APtr a;
  SplitPtr split;
  battery::vector<BranchType, Allocator> path;
  A::Snapshot root;
  BDec is_at_bot;

public:
  CUDA SearchTree(AType uid, APtr a, SplitPtr split)
   : uid_(uid), a(std::move(a)), split(std::move(split)),
     root(this->a->snapshot()), is_at_bot(BDec::bot())
  {}

  CUDA AType uid() const {
    return uid_;
  }

  CUDA BInc is_top() const {
    return BInc(lnot(is_at_bot).guard() && path.empty());
  }

  CUDA BDec is_bot() const {
    return is_at_bot;
  }

  template <class F>
  CUDA thrust::optional<TellType> interpret(const F& f) {
    return a->interpret(f);
  }

  CUDA this_type& tell(TellType&& t, BInc& has_changed) {
    a->restore(root);
    a->tell(std::move(t), has_changed);
    root = a->snapshot();
    replay();
    return *this;
  }

  /** The refinement of `a` and `split` is not done here, and if needed, must be done before calling this method.
   * This refinement operator computes \f$ \mathit{pop} \circ \mathit{push} \circ \mathit{split} \f$.
   * It initializes `a` to the next node of the search tree.
   * Therefore, `a` can backtrack, hence does not always evolve extensively and monotonically.
   * Nevertheless, the refinement operator of the search tree abstract domain is extensive and monotonic (if split is). */
  CUDA void refine(BInc& has_changed) {
    // We backtrack if no branch was pushed, it means we reached a leaf.
    pop(!push(split->split()), has_changed);
  }

private:
  CUDA void pop(bool backtrack, BInc& has_changed) {
    if(!backtrack) {
      a->tell(path.back().next(), has_changed);
    }
    else {
      while(!path.empty() && !path.back().has_next()) {
        path.pop_back();
      }
      if(!path.empty()) {
        has_changed.tell(BInc::top());
        path.back().next();
        a->restore(root);
        replay(has_changed);
      }
    }
  }

  CUDA void replay(BInc& has_changed) {
    for(int i = 0; i < path.size(); ++i) {
      a->tell(path[i].current(), has_changed);
    }
  }

  /** \return `true` if a new branch was pushed, `false` otherwise. */
  CUDA bool push(BranchType&& branch) {
    is_at_bot.tell(BDec::top());
    if(branch.size() > 0) {
      path.push_back(std::move(branch));
      return true;
    }
    return false;
  }
};

}

#endif
