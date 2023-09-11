@require_once "std/logical.c3"
@require_once "std/type_traits.c3"
@require_once "std/util.c3"

function main : () -> int = {
    if !value<AreEquivalent<int&, Ref<int>>>() {
        return 1;
    }

    if !value<AreEquivalent<int&, Ref<int&>>>() {
        return 2;
    }

    if value<AreEquivalent<int, Ref<int>>>() {
        return 3;
    }
    return 0;
}

/// @COMPILE
/// @RUN
