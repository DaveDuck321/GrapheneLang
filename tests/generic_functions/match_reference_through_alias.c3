typedef [T] _SpanImpl : { data : T, length : isize }

typedef [T] View : _SpanImpl<T[&]>
typedef [T] Span : _SpanImpl<T[mut&]>

function [T] length : (span : _SpanImpl<T>&) -> isize = {
    return span.length;
}

function main : () -> int = {
    mut data : int[2] = {0, 1};

    let span : Span<int> = {&mut data, 2};
    let view : View<int> = {&data, 2};
    return __builtin_narrow<int>(span:length());
}

/// @COMPILE
/// @RUN; EXPECT 2
