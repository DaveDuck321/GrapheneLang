@require_once "algorithms.c3"
@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "iterators.c3"
@require_once "memory.c3"
@require_once "span.c3"

typedef [T] Vector : {
    allocator : Allocator mut&,
    memory : Span<T>,
    length : isize,
}

function DEFAULT_VECTOR_CAPACITY : () -> isize = { return 8; }

function [T] make<Vector<T>> : (allocator : Allocator mut&) -> Vector<T> = {
    // HACK we don't want to allocate by default, so we are forced to initialize
    // the Span with a null pointer. Note that if a user calls data() before
    // any memory is allocated, then they'll get a 0-length span with a null
    // pointer.
    return {
        .allocator = &mut allocator,
        .memory = { &mut __builtin_int_to_ptr<T[mut&]>(0), 0 },
        .length = 0,
    };
}

function [T] push_back : (self : Vector<T> mut&, data : T) -> void = {
    if self.length == self.memory.length {
        // Grow the vector
        let new_capacity : isize = max(2 * self.memory.length, DEFAULT_VECTOR_CAPACITY());
        let new_memory : Span<T> = self.allocator:allocate_span<T>(new_capacity);

        // Copy the old data
        // TODO: use a memcpy/ for loop
        mut index : isize = 0;
        while index < self.length {
            new_memory.data[index] = self.memory.data[index];
            index += 1;
        }

        self:deallocate();
        self.memory = new_memory;
    }
    self.memory.data[self.length] = data;

    self.length += 1;
}

function [T, Iter] extend_back : (self : Vector<T> mut&, iter : Iter) -> void = {
    // TODO: create capacity in one go
    for value in iter {
        self:push_back(value);
    }
}

function [T] pop_back : (self : Vector<T> mut&) -> T = {
    runtime_assert(self.length > 0);

    self.length -= 1;
    return self.memory.data[self.length];
}

function [T] get : (self : Vector<T>&, index : isize) -> T mut& = {
    runtime_assert(index >= 0 and index < self.length);
    return &mut self.memory.data[index];
}

function [T] get : (self : Vector<T> mut&, index : isize) -> T mut& = {
    runtime_assert(index >= 0 and index < self.length);
    return &mut self.memory.data[index];
}

function [T] length : (self : Vector<T>&) -> isize = {
    return self.length;
}

function [T] length : (self : Vector<T> mut&) -> isize = {
    // TODO: I really don't like this way of supporting UFCS
    return self.length;
}

function [T] capacity : (self : Vector<T>&) -> isize = {
    return self.memory.length;
}

function [T] capacity : (self : Vector<T> mut&) -> isize = {
    return self.memory.length;
}

function [T] data : (self : Vector<T> mut&) -> Span<T> = {
    return self.memory:slice_to(self.length);
}

function [T] data : (self : Vector<T>&) -> Span<T> = {
    // TODO: we can extract the mutable data from a constant vector... Do I like this?
    return self.memory:slice_to(self.length);
}

function [T] view : (self : Vector<T> mut&) -> View<T> = {
    // TODO: take value types as const reference
    let data : Span<T> = self:data();
    return data:to_view();
}

function [T] view : (self : Vector<T>&) -> View<T> = {
    // TODO: take value types as const reference
    let data : Span<T> = self:data();
    return data:to_view();
}

function [T] get_iter : (vec : Vector<T>&) -> ContiguousContainerIter<Vector<T>&, T mut&> = {
    return { .container = &vec, .current_index = 0 };
}

function [T] get_citer : (vec : Vector<T>&) -> ContiguousContainerIter<Vector<T>&, T&> = {
    return { .container = &vec, .current_index = 0 };
}

function [T] clear : (self : Vector<T> mut&) -> void = {
    // TODO: destructor support (if we add destructors)
    self.length = 0;
}

function [T] deallocate : (self : Vector<T> mut&) -> void = {
    // TODO: we're managing to deallocate a immutable vector (I've made it mutable for aesthetics tho)
    // TODO: this aught to be a destructor
    if self.memory.length != 0 {
        // BUG deallocating a null pointer segfaults.
        self.allocator:deallocate_span(self.memory);
    }
}
