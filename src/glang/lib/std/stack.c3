@require_once "vector.c3"

typedef [T] Stack : {
    data : Vector<T>
}

function [T] make<Stack<T>> : (allocator : Allocator mut&) -> Stack<T> = {
    return { make<Vector<T>>(&mut allocator) };
}

function [T] push : (self : Stack<T> mut&, value : T) -> void = {
    self.data:push_back(value);
}

function [T] pop : (self : Stack<T> mut&) -> T = {
    return self.data:pop_back();
}

function [T] top : (self : Stack<T> mut&) -> T = {
    return self.data:get(self.data:length() - 1);
}

function [T] length : (self : Stack<T> mut&) -> isize = {
    return self.data:length();
}

function [T] empty : (self : Stack<T> mut&) -> bool = {
    return self:length() == 0;
}

function [T] deallocate : (self : Stack<T> mut&) -> void = {
    self.data:deallocate();
}
