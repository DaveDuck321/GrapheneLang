#!/usr/bin/env bash

# Atm this is easy... it becomes harder once we remove lark
./glang ./parser/parser.c3 -o ./parser/parser --bootstrap -O3
