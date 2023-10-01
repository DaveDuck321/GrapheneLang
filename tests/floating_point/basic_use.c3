@require_once "std/arithmetic.c3"

function main : () -> int = {
    // Check that llvm doesn't complain about hard-to-represent floats
    let a : f32 = 0.3;
    let b : f32 = 1.111111111111111111111111111111111111;
    let c : f32 = 340282346638528859811704183484516925440.0;
    let d : f32 = 1.17549449095e-38;  // BUG: 1.17549435082e-38 should be possible

    // Check that we can parse everything
    let e : f32 = 1.0;
    let f : f32 = 1e30;
    let g : f32 = 1e+9;
    let h : f32 = 0.0;
    return 0;
}

/// @COMPILE
/// @RUN
