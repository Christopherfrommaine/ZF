operatorTokens = '=¬∧∨∀∃⇔⇒∅∈&^|'
print('here are some helpful charachters for building logical statements: \n ' + operatorTokens)


def removeAll(el, l):
    for _ in range(l.count(el)):
        l.remove(el)


class op:
    def __init__(self, val1, val2):
        self.v1 = val1
        self.v2 = val2
        self.opSymbol = None

    def wff(self):
        return f'({self.v1.wff()}){self.opSymbol}({self.v2.wff()})'

    def eqWFF(self, other):
        return self.wff() == other.wff()

    def eqNum(self, other):
        pass

    def __repr__(self):
        return f'({self.v1}){self.opSymbol}({self.v2})'
        # return f'{type(self)}({self.v1}, {self.v2})'

    def __invert__(self):
        return opNot(self)

    def __or__(self, other):
        return opOr(self, other)

    def __and__(self, other):
        return opAnd(self, other)

    def __ror__(self, other):
        return self.__or__(other)

    def __rand__(self, other):
        return self.__and__(other)
class opNot(op):
    def __init__(self, val1):
        op.__init__(self, val1, None)
        self.opSymbol = '¬'

    def __repr__(self):
        return f'{self.opSymbol}({self.v1})'
class opOr(op):
    def __init__(self, val1, val2):
        op.__init__(self, val1, val2)
        self.opSymbol = '∨'
class opAnd(op):
    def __init__(self, val1, val2):
        op.__init__(self, val1, val2)
        self.opSymbol = '∧'
class opVar(op):
    def __init__(self, name, value=None):
        op.__init__(self, None, None)
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name


def tokenizer(string):
    o = []
    temp = ''
    parensDepth = 0
    for char in string:
        if not parensDepth and char in operatorTokens:
            o.append(temp)
            o.append(char)
            temp = ''
        else:
            temp += char

        if char == '(':
            temp = ''
            parensDepth += 1
        if char == ')':
            parensDepth -= 1
            if not parensDepth:
                o.append(tokenizer(temp[:-1]))
                temp = ''
    o.append(temp)
    removeAll('', o)
    return o

def parser(tokens):
    # Parens
    tokens = [parser(tok) if isinstance(tok, list) else tok for tok in tokens]
    
    # Not
    index = 0
    cont = True
    while cont:
        cont = False
        for i, tok in enumerate(tokens):
            if tok == '¬':
                tokens = tokens[:i] + [opNot(tokens[i + 1])] + tokens[i + 2:]


    return tokens


test_L = '¬p∧¬q⇔¬(p∨q)'

T = tokenizer(test_L)
print(T)
print(parser(T))


class statement:
    def __init__(self, stringRepr='T'):
        self.str = stringRepr

        self.variables = []
        self.ast = None
