class Var:

    def __init__(self, value=None, operation_var=None):
        self.value = value
        self.operation_var = operation_var or [self]
        self.name = "var" + str(id(self))

    def __add__(self, other):
        var = Add(operation_var=[self, other])
        return var

    def __mul__(self, other):
        var = Mul(operation_var=[self, other])
        return var

    def __sub__(self, other):
        var = Sub(operation_var=[self, other])
        return var

    def __truediv__(self, other):
        var = TrueDiv(operation_var=[self, other])
        return var

    def __repr__(self):
        return str(self.forward())

    def forward(self):
        return self.value

    def gradient(self, other):
        if self.name == other.name:
            return 1
        else:
            return 0


class Add(Var):
    def __init__(self, value=None, operation_var=None):
        super().__init__(value=value, operation_var=operation_var)

    def __call__(self, cls, other):
        self.__init__(operation_var=[cls, other])

    def forward(self):
        self.value = self.operation_var[0].forward() + self.operation_var[1].forward()
        return self.value

    def gradient(self, other):
        if other not in self.operation_var:
            raise ValueError
        return 1


class Sub(Var):
    def __init__(self, value=None, operation_var=None):
        super().__init__(value=value, operation_var=operation_var)

    def __call__(self, cls, other):
        self.__init__(operation_var=[cls, other])

    def forward(self):
        self.value = self.operation_var[0].forward() - self.operation_var[1].forward()
        return self.value

    def gradient(self, other):
        if other not in self.operation_var:
            raise ValueError
        if other.name == self.operation_var[0].name:
            return 1
        else:
            return -1


class TrueDiv(Var):
    def __init__(self, value=None, operation_var=None):
        super().__init__(value=value, operation_var=operation_var)

    def __call__(self, cls, other):
        self.__init__(operation_var=[cls, other])

    def forward(self):
        self.value = self.operation_var[0].forward() / self.operation_var[1].forward()
        return self.value

    def gradient(self, other):
        if other not in self.operation_var:
            raise ValueError
        if other.name == self.operation_var[0].name:
            return 1 / self.operation_var[1].forward()
        else:
            return - self.operation_var[0].forward() * self.operation_var[1].forward() ** -2


class Mul(Var):
    def __init__(self, value=None, operation_var=None):
        super().__init__(value=value, operation_var=operation_var)

    def __call__(self, cls, other):
        self.__init__(operation_var=[cls, other])

    def forward(self):
        self.value = self.operation_var[0].forward() * self.operation_var[1].forward()
        return self.value

    def gradient(self, other):
        if other not in self.operation_var:
            raise ValueError
        if other.name == self.operation_var[0].name:
            return self.operation_var[1].forward()
        else:
            return self.operation_var[0].forward()


def gradient(_input, _output):
    if _input in _output.operation_var:
        return _output.gradient(_input)
    if _output.operation_var == [_output]:
        if _input == _output:
            return 1
        else:
            return 0
    else:
        left = _output.gradient(_output.operation_var[0]) * gradient(_input, _output.operation_var[0])
        right = _output.gradient(_output.operation_var[1]) * gradient(_input, _output.operation_var[1])
        return left + right

