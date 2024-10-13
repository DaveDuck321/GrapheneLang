@require_once "iterators.c3"
@require_once "type_traits.c3"
@require_once "math.c3"

// Hacky: we can't have generic constant parameters, take the whole type instead
typedef [T] _SpanImpl : { data : T, length : isize }
typedef [T] View : _SpanImpl<T[&]>
typedef [T] Span : _SpanImpl<T[mut&]>

typedef [T] _SpanImplMemberReference<T[&]> : T&
typedef [T] _SpanImplMemberReference<T[mut&]> : T mut&

function [T] to_view : (span : Span<T>) -> View<T> = {
    return make<View<T>>(&span.data, span.length);
}

function [T] to_view : (span : Span<T>&) -> View<T> = {
    return make<View<T>>(&span.data, span.length);
}

function [T, @Len] make<View<T>> : (data : T[@Len]&) -> View<T> = {
    return {
        .data = &data,
        .length = @Len,
    };
}

function [T] make<View<T>> : (data : T[&], length : isize) -> View<T> = {
    return {
        .data = &data,
        .length = length,
    };
}

function [T, @Len] make<Span<T>> : (data : T[@Len] mut&) -> Span<T> = {
    return {
        .data = &mut data,
        .length = @Len,
    };
}

function [T] make<Span<T>> : (data : T[mut&], length : isize) -> Span<T> = {
    return {
        .data = &mut data,
        .length = length,
    };
}

function [T] get_iter : (span : _SpanImpl<T>&)
    -> ContiguousContainerIter<_SpanImpl<T>, _SpanImplMemberReference<T>> =
{
    return {
        .container = span,
        .current_index = 0,
    };
}

function [T] get_iter : (span : _SpanImpl<T> mut&)
    -> ContiguousContainerIter<_SpanImpl<T>, _SpanImplMemberReference<T>> =
{
    return {
        .container = span,
        .current_index = 0,
    };
}

function [T] length : (span : _SpanImpl<T>&) -> isize = {
    return span.length;
}

function [T] length : (span : _SpanImpl<T> mut&) -> isize = {
    return span.length;
}


function [T] get : (span : _SpanImpl<T>&, index : isize)
    -> TypeIf<_SpanImplMemberReference<T>, IsConstRef<T>> =
{
    runtime_assert(index >= 0 and index < span.length);
    return &span.data[index];
}

function [T] get : (span : _SpanImpl<T>&, index : isize)
    -> TypeIf<_SpanImplMemberReference<T>, IsMutRef<T>> =
{
    return &mut span.data[index];
}

@operator [T] == : (lhs : _SpanImpl<T>, rhs : _SpanImpl<T>) -> bool = {
    if lhs:length() != rhs:length() {
        return false;
    }

    if have_same_addr(&lhs.data, &rhs.data) {
        // Short circuit when the memory overlaps
        return true;
    }

    for i in range(lhs.length) {
        if lhs.data[i] != rhs.data[i] {
            return false;
        }
    }
    return true;
}

@operator [T] < : (lhs : _SpanImpl<T>, rhs : _SpanImpl<T>) -> bool = {
    // Use lexicographic ordering:
    // - Given two sequences of the same length, the order of the two sequences
    //   depends on the order of the symbols in the first place where the two
    //   sequences differ.
    // - If the sequences have different lengths, the shortest sequence is
    //   padded with symbols that compare less than all other symbols until the
    //   two sequences have the same length.
    let common_length : isize = min(lhs.length, rhs.length);
    for i in range(common_length) {
        if lhs:get(i) != rhs:get(i) {
            // We've found the first place where they differ.
            return lhs:get(i) < rhs:get(i);
        }
    }

    // Spans are equal.
    if lhs.length == rhs.length {
        return false;
    }

    // The shortest sequence compares less.
    return lhs.length < rhs.length;
}

function [T] slice_to : (span : Span<T>&, end : isize) -> Span<T> = {
    return make<Span<T>>(&mut span.data, min(end, span.length));
}

function [T] slice_from : (span : Span<T>&, start : isize) -> Span<T> = {
    start = clamp<isize>(0, start, span.length);
    let start_addr : iptr = ref_to_addr(&span:get(start));
    return make<Span<T>>(
        &mut addr_to_mut_heap_array<T>(start_addr),
        span.length - start
    );
}

function [T] slice_between : (span : Span<T>&, start : isize, end : isize) -> Span<T> = {
    start = clamp<isize>(0, start, span.length);
    end = clamp<isize>(0, end, span.length);
    let start_addr : iptr = ref_to_addr(&span:get(start));
    return make<Span<T>>(
        &mut addr_to_mut_heap_array<T>(start_addr),
        end - start
    );
}


// TODO: this duplication is nasty (and largely unavoidable without language features)
function [T] slice_to : (view : View<T>&, end : isize) -> View<T> = {
    return make<View<T>>(&view.data, min(end, view.length));
}

function [T] slice_from : (view : View<T>&, start : isize) -> View<T> = {
    start = clamp<isize>(0, start, view.length);
    let start_addr : iptr = ref_to_addr(&view:get(start));
    return make<View<T>>(
        &addr_to_heap_array<T>(start_addr),
        view.length - start
    );
}

function [T] slice_between : (view : View<T>&, start : isize, end : isize) -> View<T> = {
    start = clamp<isize>(0, start, view.length);
    end = clamp<isize>(0, end, view.length);
    let start_addr : iptr = ref_to_addr(&view:get(start));
    return make<View<T>>(
        &addr_to_heap_array<T>(start_addr),
        end - start
    );
}
