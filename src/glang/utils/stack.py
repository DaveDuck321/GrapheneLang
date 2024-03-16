class Stack[T]:
    def __init__(self) -> None:
        self._lst: list[T] = []

    def push(self, value: T) -> None:
        self._lst.append(value)

    def pop(self) -> T:
        if self._lst:
            return self._lst.pop()
        raise IndexError

    def peek(self) -> T:
        if self._lst:
            return self._lst[-1]
        raise IndexError

    def empty(self) -> bool:
        return not self._lst
