@require_once "std/type_traits.c3"

@operator[T, Indicies...] [] : (array: T[&], indicies : Indicies...) -> ConstRef<T> = {
    return &__builtin_array_index(&array, indicies...);
}

@operator[T, @N, Indicies...] [] : (array: T[@N]&, indicies : Indicies...) -> ConstRef<T> = {
    return &__builtin_array_index(&array, indicies...);
}

@operator[T, Indicies...] [] : (array: T[mut&], indicies : Indicies...) -> MutRef<T> = {
    return &mut __builtin_array_index(&mut array, indicies...);
}

@operator[T, @N, Indicies...] [] : (array: T[@N] mut&, indicies : Indicies...) -> MutRef<T> = {
    return &mut __builtin_array_index(&mut array, indicies...);
}
