function a : (x: i32, y: i64) -> int = { return 0; };
function a : (x: i64, y: i32) -> int = { return 1; };

function main: () -> int = {
    return a(0, 0);
};
