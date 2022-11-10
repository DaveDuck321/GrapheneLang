from lark import Lark

with open("grammar.lark") as grammar:
    l = Lark(grammar, parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())
