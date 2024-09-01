@require_once "std/type_traits.c3"
@require_once "std/assert.c3"

function main : () -> int = {
    runtime_assert(value<AreEquivalent<ConstRef<int>, int&>>());
    runtime_assert(value<AreEquivalent<ConstRef<int&>, int&>>());
    runtime_assert(value<AreEquivalent<ConstRef<int mut&>, int&>>());
    runtime_assert(value<AreEquivalent<ConstRef<int[&]>, int[&]>>());
    runtime_assert(value<AreEquivalent<ConstRef<int[4, 3, 2]>, int[4, 3, 2]&>>());
    runtime_assert(value<AreEquivalent<ConstRef<int[mut&]>, int[&]>>());

    runtime_assert(value<AreEquivalent<MutRef<int>, int mut&>>());
    runtime_assert(value<AreEquivalent<MutRef<int&>, int mut&>>());
    runtime_assert(value<AreEquivalent<MutRef<int mut&>, int mut&>>());
    runtime_assert(value<AreEquivalent<MutRef<int[&]>, int[mut&]>>());
    runtime_assert(value<AreEquivalent<MutRef<int[4, 3, 2]>, int[4, 3, 2] mut&>>());
    runtime_assert(value<AreEquivalent<MutRef<int[mut&]>, int[mut&]>>());

    runtime_assert(!value<AreEquivalent<MutRef<int>, int&>>());
    return 0;
}

/// @COMPILE
/// @RUN
