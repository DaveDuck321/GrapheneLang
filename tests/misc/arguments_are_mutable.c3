function main : (argc : int, argv : u8[&][&]) -> int = {
    // We test two things
    // - argc is mutable
    // - argv is immutable (tested implicitly, otherwise this wouldn't compile)
    argc = 1;
    return 0;
}

/// @COMPILE
/// @RUN
