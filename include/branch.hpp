// Copyright 2022 Pierre Talbot

#ifndef BRANCH_HPP
#define BRANCH_HPP

#include "ast.hpp"
#include "vector.hpp"

namespace lala {

template <class TellTy, class Alloc>
class Branch {
public:
  using TellType = TellTy;
  using Allocator = Alloc;

private:
  battery::vector<TellType, Allocator> children;
  int current;

public:
  CUDA Branch(): children(), current(-1) {}
  CUDA Branch(Branch&&) = default;
  CUDA Branch(battery::vector<TellType, Allocator> &&children)
   : children(std::move(children)), current(-1) {}

  CUDA int size() const {
    return children.size();
  }

  CUDA const TellType& next() {
    assert(has_next());
    return children[++current];
  }

  CUDA bool has_next() const {
    return current + 1 < children.size();
  }

  CUDA bool is_pruned() const {
    return current >= children.size();
  }

  CUDA const TellType& current() const {
    assert(!is_pruned() && current != -1);
    return children[current];
  }
};

}

#endif
