from lark import Lark

with open("grammar.lark") as grammar:
    l = Lark(grammar)

print(l.parse("""typedef foo: (int, int) -> int;"""))
