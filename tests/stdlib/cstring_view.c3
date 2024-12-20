@require_once "std/string.c3"
@require_once "std/assert.c3"

function main : () -> int = {
    let foo : CStringView = make<CStringView>("foo\0");
    let cstr : CString = make<CString>("bar\0");
    let bar : CStringView = make<CStringView>(cstr);

    runtime_assert(foo:length() == 3);
    runtime_assert(bar:length() == 3);
    runtime_assert(foo != bar);
    runtime_assert(foo:slice_from(2) == make<CStringView>("o\0"));
    runtime_assert(foo:slice_from(4) == make<CStringView>("\0"));

    runtime_assert(make<CStringView>(&foo:data()) == foo);

    return 0;
}

/// @COMPILE
/// @RUN
