// Copyright 2025 Pierre Talbot

#ifndef LALA_POWER_LIGHT_BRANCH_HPP
#define LALA_POWER_LIGHT_BRANCH_HPP

/** Similar to `Branch` but specialized to binary search tree splitting over universe (e.g. interval). */

namespace lala {

template <class U>
struct LightBranch {
  template <class U2>
  friend class LightBranch;

  AVar var;
  U children[2];
  int current_idx;

  CUDA INLINE LightBranch(): current_idx(-1) {}
  LightBranch(const LightBranch&) = default;
  LightBranch(LightBranch&&) = default;
  CUDA INLINE LightBranch(AVar var, const U& left, const U& right)
   : var(var), current_idx(-1)
  {
    children[0] = left;
    children[1] = right;
  }

  CUDA INLINE const U& next() {
    assert(has_next());
    return children[++current_idx];
  }

  CUDA INLINE const U& operator[](int idx) {
    return children[idx];
  }

  CUDA INLINE bool has_next() const {
    return current_idx < 1;
  }

  CUDA INLINE void prune() {
    current_idx = 2;
  }

  CUDA INLINE bool is_pruned() const {
    return current_idx >= 2;
  }

  CUDA INLINE const U& current() const {
    assert(current_idx != -1 && current_idx < 2);
    return children[current_idx];
  }
};

}

#endif
