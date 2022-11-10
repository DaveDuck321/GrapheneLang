from lark import Lark

l = Lark.open("grammar.lark", parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())
