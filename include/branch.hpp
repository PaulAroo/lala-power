// Copyright 2022 Pierre Talbot

#ifndef BRANCH_HPP
#define BRANCH_HPP

#include "logic/logic.hpp"
#include "vector.hpp"

namespace lala {

template <class TellTy, class Alloc>
class Branch {
public:
  using tell_type = TellTy;
  using allocator_type = Alloc;

private:
  battery::vector<tell_type, allocator_type> children;
  int current_idx;

public:
  CUDA Branch(): children(), current_idx(-1) {}
  Branch(const Branch&) = default;
  Branch(Branch&&) = default;
  CUDA Branch(battery::vector<tell_type, allocator_type>&& children)
   : children(std::move(children)), current_idx(-1) {}

  CUDA int size() const {
    return children.size();
  }

  CUDA const tell_type& next() {
    assert(has_next());
    return children[++current_idx];
  }

  CUDA bool has_next() const {
    return current_idx + 1 < size();
  }

  CUDA void prune() {
    current_idx = size();
  }

  CUDA bool is_pruned() const {
    return current_idx >= size();
  }

  CUDA const tell_type& current() const {
    assert(!is_pruned() && current_idx != -1 && current_idx < children.size());
    return children[current_idx];
  }
};

}

#endif
