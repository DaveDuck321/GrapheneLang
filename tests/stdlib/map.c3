@require_once "std/map.c3"
@require_once "std/format.c3"
@require_once "std/assert.c3"
@require_once "std/string.c3"

function main : () -> int = {
    mut map : Map<int, StringView> = make<Map<int, StringView>>();

    runtime_assert(map:empty() == true);
    runtime_assert(map:size() == 0);

    map:insert(0, sv("0"));
    map:insert(1, sv("1"));
    map:insert(2, sv("2"));
    map:insert(3, sv("3"));

    runtime_assert(map:contains(0));
    runtime_assert(map:contains(1));
    runtime_assert(map:contains(2));
    runtime_assert(map:contains(3));

    runtime_assert(map:empty() == false);
    runtime_assert(map:size() == 4);

    print(map:at(4));
    print(map:at(3));
    print(map:at(2));
    print(map:at(1));
    print(map:at(0));

    for i in map:get_citer() {
        printf(sv("% = %"), i.key, i.value);
    }

    runtime_assert(map:erase(0));
    runtime_assert(!map:erase(4));

    runtime_assert(!map:contains(0));
    runtime_assert(!map:contains(4));

    runtime_assert(map:empty() == false);
    runtime_assert(map:size() == 3);

    map:clear();

    runtime_assert(map:empty() == true);
    runtime_assert(map:size() == 0);

    print(map:at(4));
    print(map:at(3));
    print(map:at(2));
    print(map:at(1));
    print(map:at(0));

    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// <empty>
/// 3
/// 2
/// 1
/// 0
/// 0 = 0
/// 1 = 1
/// 2 = 2
/// 3 = 3
/// <empty>
/// <empty>
/// <empty>
/// <empty>
/// <empty>
