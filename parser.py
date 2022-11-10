from lark import Lark

with open("grammar.lark") as grammar:
    l = Lark(grammar, parser="lalr", start="program")

print(l.parse("""typedef foo: (int, int) -> int;""").pretty())
