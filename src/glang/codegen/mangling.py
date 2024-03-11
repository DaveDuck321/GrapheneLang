from collections.abc import Iterable

from glang.codegen.interfaces import SpecializationItem, Type


def mangle_int(size_in_bits: int, *, is_signed: bool) -> str:
    # <builtin-type> ::= c  # char
    #                ::= a  # signed char
    #                ::= h  # unsigned char
    #                ::= s  # short
    #                ::= t  # unsigned short
    #                ::= i  # int
    #                ::= j  # unsigned int
    #                ::= l  # long
    #                ::= m  # unsigned long
    #                ::= x  # long long, __int64
    #                ::= y  # unsigned long long, __int64
    #                ::= n  # __int128
    #                ::= o  # unsigned __int128
    match size_in_bits, is_signed:
        case 8, True:
            return "c"
        case 8, False:
            return "h"
        case 16, True:
            return "s"
        case 16, False:
            return "t"
        case 32, True:
            return "i"
        case 32, False:
            return "j"
        case 64, True:
            return "l"
        case 64, False:
            return "m"
        case 128, True:
            return "n"
        case 128, False:
            return "o"

    raise AssertionError


def mangle_float(size_in_bits: int) -> str:
    # <builtin-type> ::= DF <number> _  # _FloatN (N bits)
    #                ::= f              # float
    #                ::= d              # double
    #                ::= g              # __float128
    match size_in_bits:
        case 16:
            return "DF16_"
        case 32:
            return "f"
        case 64:
            return "d"
        case 128:
            return "g"

    raise AssertionError


def mangle_source_name(identifier: str) -> str:
    assert len(identifier) > 0

    # <source-name> ::= <positive length number> <identifier>
    return f"{len(identifier)}{identifier}"


def mangle_template_arg(arg: SpecializationItem) -> str:
    if isinstance(arg, Type):
        # <template-arg> ::= <type>  # type or template
        return arg.ir_mangle
    else:
        assert isinstance(arg, int)
        # <template-arg> ::= <expr-primary>             # simple expressions
        # <expr-primary> ::= L <type> <value number> E  # integer literal
        # FIXME mangle as an int, even though we don't actually force a type.
        # TODO negative numbers.
        return f"Li{arg}E"


def mangle_template_args(args: Iterable[SpecializationItem]) -> str:
    # <template-args> ::= I <template-arg>+ E
    return f"I{str.join('', map(mangle_template_arg, args))}E"


def mangle_operator_name(name: str) -> str:
    # <operator-name> ::= co  # ~
    #                 ::= pl  # +
    #                 ::= mi  # -
    #                 ::= ml  # *
    #                 ::= dv  # /
    #                 ::= rm  # %
    #                 ::= an  # &
    #                 ::= or  # |
    #                 ::= eo  # ^
    #                 ::= pL  # +=
    #                 ::= mI  # -=
    #                 ::= mL  # *=
    #                 ::= dV  # /=
    #                 ::= rM  # %=
    #                 ::= aN  # &=
    #                 ::= oR  # |=
    #                 ::= eO  # ^=
    #                 ::= ls  # <<
    #                 ::= rs  # >>
    #                 ::= lS  # <<=
    #                 ::= rS  # >>=
    #                 ::= eq  # ==
    #                 ::= ne  # !=
    #                 ::= lt  # <
    #                 ::= gt  # >
    #                 ::= le  # <=
    #                 ::= ge  # >=
    #                 ::= ss  # <=>
    #                 ::= nt  # !
    #                 ::= ix  # []
    #                 ::= v <digit> <source-name>	# vendor extended operator
    match name:
        case "~":
            return "co"
        case "+":
            return "pl"
        case "-":
            return "mi"
        case "*":
            return "ml"
        case "/":
            return "dv"
        case "%":
            return "rm"
        case "&":
            return "an"
        case "|":
            return "or"
        case "^":
            return "eo"
        case "+=":
            return "pL"
        case "-=":
            return "mI"
        case "*=":
            return "mL"
        case "/=":
            return "dV"
        case "%=":
            return "rM"
        case "&=":
            return "aN"
        case "|=":
            return "oR"
        case "^=":
            return "eO"
        case "<<":
            return "ls"
        case ">>":
            return "rs"
        case "<<=":
            return "lS"
        case ">>=":
            return "rS"
        case "==":
            return "eq"
        case "!=":
            return "ne"
        case "<":
            return "lt"
        case ">":
            return "gt"
        case "<=":
            return "le"
        case ">=":
            return "ge"
        case "<=>":
            return "ss"
        case "!":
            return "nt"
        case "[]":
            return "ix"
        # Vendors who define builtin extended operators (e.g. __imag) shall
        # encode them as a 'v' prefix followed by the operand count as a single
        # decimal digit, and the name in <length,ID> form.
        case "**":
            return "v2" + mangle_source_name("power")
        case "@":
            return "v2" + mangle_source_name("matmul")

    raise AssertionError(name)
