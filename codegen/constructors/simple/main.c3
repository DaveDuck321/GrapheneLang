typedef my_struct : { a: int, b: int };

@constructor : (this: my_struct&) -> () = {
    this.a = 1;
    this.b = 2;
};

@constructor : (this: my_struct&, a_: int) -> () = {
    this.a = a_;
    this.b = 2;
};

@constructor : (this: my_struct&, a_: int, b_: int) -> () = {
    this.a = a_;
    this.b = b_;
};

function main : () -> int = {
    let s : my_struct = {}; // Should call the default constructor.
    let s : my_struct = { 4 };
    let s : my_struct = { 4, 5 };
};
