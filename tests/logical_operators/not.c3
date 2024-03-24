function main : () -> int = {
    if not true != false {
        return 1;
    }

    if not false != true {
        return 2;
    }

    return 0;
}

/// @COMPILE
/// @RUN
