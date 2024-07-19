operatorTokens = '⚠¬!∧&∨|⊻^∀∃⇒⇔='  # In order of precedence, must keep the error symbol at the beginning, as it is the symbol for variables, which have precedence over everything else.
whitespaceTokens = ' \n'
breakOnChars = operatorTokens + whitespaceTokens + '()[]{}'
operatorPrecedence = {opSymbol: i for i, opSymbol in enumerate(operatorTokens)} 

print('here are some helpful charachters for building logical statements: \n ' + operatorTokens)

class op:
    def __init__(self, val1, val2):
        self.v1 = val1
        self.v2 = val2

    arity = 2
    symbol = '⚠'

    def __repr__(self):
        inParens = lambda v: f'({v})' if operatorPrecedence[type(self).symbol] < operatorPrecedence[type(v).symbol] else f'{v}'
        match type(self).arity:
            case 1:
                return type(self).symbol + inParens(self.v1)
            case 2:
                return inParens(self.v1) + type(self).symbol + inParens(self.v2)
            case _:
                raise Exception('wrong arity')
   
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
    
    symbol = '¬'
    arity = 1

class opOr(op):
    symbol = '∨'
class opAnd(op):
    symbol = '∧'
class opXor(op):
    symbol = '⊻'
class opFA(op):
    symbol = '∀'
class opTE(op):
    symbol = '∃'
class opIf(op):
    symbol = '⇒'
class opImp(op):
    symbol = '⇔'
class opVar(op):
    def __init__(self, name, value=None):
        op.__init__(self, None, None)
        self.name = name
        self.value = value

        if self.name == 'True':
            self.value = True
        if self.name == 'False':
            self.value = False
    arity = None
    symbol = '⚠'

    def __repr__(self):
        return self.name

opFromTok = {'¬': opNot, '!': opNot, '∧': opAnd, '&': opAnd, '∨': opOr, '|': opOr, '⊻': opXor, '^': opXor, '∀': opFA, '∃': opTE, '⇒': opIf, '⇔': opImp, '=': opImp} 


def tokenize(string):
    o = []
    temp = ''
    for char in string:
        if char in breakOnChars:
            if temp != '':
                o.append(temp)
                temp = ''
            if char not in whitespaceTokens:
                o.append(char)
        else:
            temp += char
    if temp:
        o.append(temp)
    return o

def shuntingYard(tokens):
    stack = []
    queue = []
    for tok in tokens:
        if tok in operatorTokens:
            while stack and stack[-1] in operatorTokens and operatorPrecedence[stack[-1]] < operatorPrecedence[tok]:
                queue.append(stack.pop())
            stack.append(tok)
        elif tok == '(':
            stack.append('(')
        elif tok == ')':
            while stack[-1] != '(':
                queue.append(stack.pop())
            stack.pop()
        else:
            queue.append(tok)
    queue += reversed(stack)
    return queue


def ASTfromRP(tokens):
    solveStack = []
    for tok in tokens:
        if tok in operatorTokens:
            opType = opFromTok[tok]
            solveStack = solveStack[:-opType.arity] + [opType(*solveStack[-opType.arity:])]
        else:
            solveStack.append(opVar(tok))
    if len(solveStack) != 1:
        raise Exception("Too few operators")
    return solveStack[0]


test_L = '¬p∧¬q⇔¬(p∨q)'
test_tokens = tokenize(test_L)
print(test_tokens)

test_L = ' ¬p  ∧\n¬ q        ⇔¬ (                          p \n\n\n\n∨q)'
test_tokens = tokenize(test_L)
print(test_tokens)

print(shuntingYard(test_tokens))
print(ASTfromRP(shuntingYard(test_tokens)))

