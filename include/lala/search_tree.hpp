// Copyright 2022 Pierre Talbot

#ifndef SEARCH_TREE_HPP
#define SEARCH_TREE_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/copy_dag_helper.hpp"

namespace lala {

template <class A>
class SearchTree {
public:
  using allocator_type = typename A::allocator_type;
  using branch_type = typename Split::branch_type;
  template <class Alloc>
  using ask_type = typename A::ask_type<Alloc>;
  using universe_type = typename A::universe_type;
  using sub_type = A;
  using sub_ptr = battery::shared_ptr<A, allocator_type>;
  using split_type = SplitStrategy<A>;
  using split_ptr = battery::shared_ptr<split_type, allocator_type>;
  using this_type = SearchTree<A>;

  template <class Alloc>
  struct tell_type {
    battery::vector<typename A::tell_type<Alloc>, Alloc> sub_tells;
    battery::vector<typename split_type::tell_type<Alloc>, Alloc> split_tells;
    CUDA tell_type(const Alloc& alloc): sub_tells(alloc), split_tells(alloc) {}
    CUDA tell_type(const tell_type&) = default;
  };

  template<class F, class Env>
  using iresult_tell = IResult<tell_type<typename Env::allocator_type>, F>;

  template<class F, class Env>
  using iresult_ask = typename A::iresult_ask<F, Env>;

  constexpr static const char* name = "SearchTree";

private:
  AType atype;
  // `a` reflects the current node of the search tree being refined and expanded.
  // If the search tree is `top` (i.e., empty), then `a` is equal to `nullptr`.
  sub_ptr a;
  split_ptr split;
  battery::vector<branch_type, allocator_type> stack;
  using root_type = battery::tuple<
    typename sub_type::snapshot_type<allocator_type>,
    typename split_type::snapshot_type<allocator_type>>;
  root_type root;
  // Tell formulas (and strategies) to be added to root on backtracking.
  tell_type<allocator_type> root_tell;

public:
  CUDA SearchTree(AType uid, sub_ptr a, split_ptr split)
   : atype(uid), a(std::move(a)), split(std::move(split)),
     stack(this->a->get_allocator()), root_tell(this->a->get_allocator())
  {}

  template<class A2, class Alloc2, class FastAlloc>
  CUDA SearchTree(const SearchTree<A2, Alloc2>& other, AbstractDeps<allocator_type, FastAlloc>& deps)
   : atype(other.atype), a(deps.template clone<A>(other.a)), split(deps.template clone<Split>(other.split)),
     stack(other.stack), root(other.root), root_tell(other.root_tell)
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
    // We need short-circuit using && due to `a` possibly a null pointer.
    return is_singleton() && a->is_bot();
  }

  CUDA local::BInc is_top() const {
    return !bool(a);
  }

private:
  template <class F, class Env, class Alloc>
  CUDA void interpret_tell_in(const F& f, Env& env, iresult_tell<F, Env>& res) {
    if(f.is(Seq) && f.seq().sig() == AND) {
      for(int i = 0; !res.is_error() && i < f.seq().size(); ++i) {
        interpret_tell_in(f.seq(i), env, res);
      }
    }
    else if(f.is(ESeq) && f.seq().esig() == "search") {
      auto split_res = split->interpret_tell_in(f, env);
      if(split_res.has_value()) {
        res.value().split_tells.push_back(std::move(split_res.value()));
      }
      else {
        res = iresult_tell<F, Env>(split_res.error());
      }
    }
    else {
      auto r = a->interpret_tell_in(f, env);
      if(r.has_value()) {
        res.value().sub_tells.push_back(std::move(r.value()));
      }
      else {
        res = iresult_tell<F, Env>(r.error());
      }
    }
  }

public:
  template <class F, class Env>
  CUDA iresult_tell<F, Env> interpret_tell_in(const F& f, Env& env) {
    if(is_top()) {
      return iresult<F, Env>(IError<F>(true, name, "The current abstract element is `top`.", f));
    }
    iresult_tell<F, Env> res(tell_type<allocator_type>(env.get_allocator()));
    interpret_tell_in(f, env, res);
    return res;
  }

  template <class F, class Env>
  CUDA iresult_ask<F, Env> interpret_ask_in(const F& f, Env& env) {
    return a->interpret_ask_in(f, env);
  }

private:
  template <class Alloc, class Mem>
  CUDA void tell_current(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    for(int i = 0; i < t.sub_tells.size(); ++i) {
      a->tell(t.sub_tells[i], has_changed);
    }
    for(int i = 0; i < t.split_tells.size(); ++i) {
      split->tell(t.split_tells[i], has_changed);
    }
  }
public:

  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    if(!is_top()) {
      if(!is_singleton()) {
        // We will add `t` to root when we backtrack (see `pop`) and have a chance to modify the root node.
        for(int i = 0; i < t.sub_tells.size(); ++i) {
          root_tell.sub_tells.push_back(t.sub_tells[i]);
        }
        for(int i = 0; i < t.split_tells.size(); ++i) {
          root_tell.split_tells.push_back(t.split_tells[i]);
        }
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

  /** Extract an under-approximation if the last node popped \f$ a \f$ is an under-approximation.
   * The under-approximation consists in a search tree \f$ \{a\} \f$ with a single node.
   * \pre `ua` must be different from `top`. */
  CUDA bool extract(this_type& ua) const {
    if(!is_top()) {
      assert(bool(ua.a));
      if(a->extract(*ua.a)) {
        ua.stack.clear();
        ua.root_tell.sub_tells.clear();
        ua.root_tell.split_tells.clear();
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
        root = battery::make_tuple(
          a->template snapshot<allocator_type>(),
          split->template snapshot<allocator_type>());
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
        a->template snapshot<allocator_type>(),
        split->template snapshot<allocator_type>());
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
