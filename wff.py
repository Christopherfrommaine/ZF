operatorTokens = '⚠¬!∧&∨|⊻^∀∃⇒⇔='  # In order of precedence, must keep the error symbol at the beginning, as it is the symbol for variables, which have precedence over everything else.
whitespaceTokens = ' \n'
breakOnChars = operatorTokens + whitespaceTokens + '()[]{}'
operatorPrecedence = {opSymbol: i for i, opSymbol in enumerate(operatorTokens)} 

print('here are some helpful charachters for building logical statements: \n ' + operatorTokens + '\n')

class op:
    def __init__(self, val1, val2):
        self.v1 = val1
        self.v2 = val2

    arity = 2
    symbol = '⚠'
    symmetric = True

    def __repr__(self):
        inParens = lambda v: f'({v})' if operatorPrecedence[type(self).symbol] < operatorPrecedence[type(v).symbol] else f'{v}'
        if type(self).arity == 1:
            return type(self).symbol + inParens(self.v1)
        return inParens(self.v1) + type(self).symbol + inParens(self.v2)
   
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
    
    def visualEquals(self, other):
        return str(self) == str(other)
    def recursiveEquals(self, other):
        if self.arity != other.arity:
            return False

        if self.arity == 1:
            return self.v1.recursiveEquals(other.v1)

        if self.symmetric:
            return (self.v1.recursiveEquals(other.v1) and self.v2.recursiveEquals(other.v2)) or (self.v1.recursiveEquals(other.v2) and self.v2.recursiveEquals(other.v1))
        else:
            return self.v1.recursiveEquals(other.v1) and self.v2.recursiveEquals(other.v2)

    def simplify(self):
        if self.arity is None:
            return self
        if hasattr(self, "simplifyStep"):
            o = self.simplifyStep()
            if o is not self:
                return o.simplify()
        return type(self)(self.v1.simplify()) if self.arity == 1 else type(self)(self.v1.simplify(), self.v2.simplify())

    def evaluate(self, globs=None):
        if self.arity == 1:
            return type(self)(self.v1.numEval__(globs)).simplify()
        return type(self)(self.v1.numEval__(globs), self.v2.numEval__(globs)).simplify()

    def variables(self):
        if self.arity == 1:
            return self.v1.variables()
        return set.union(self.v1.variables(), self.v2.variables())

class opNot(op):
    def __init__(self, val1):
        op.__init__(self, val1, None)
    
    symbol = '¬'
    arity = 1

    def simplifyStep(self):
        # Evaluation
        if isinstance(self.v1, opVar) and self.v1.value is not None:
            return opVar(not self.v1.value)

        # Double Negation
        if isinstance(self.v1, opNot):
            return self.v1.v1
        
        # DeMorgan's laws
        elif isinstance(self.v1, opAnd):
            if isinstance(self.v1.v1, opNot) and isinstance(self.v1.v2, opNot):
                return opOr(self.v1.v1.v1, self.v1.v2.v1)
            return opOr(opNot(self.v1.v1), opNot(self.v1.v2))
        elif isinstance(self.v1, opOr):
            if isinstance(self.v1.v1, opNot) and isinstance(self.v1.v2, opNot):
                return opAnd(self.v1.v1.v1, self.v1.v2.v1)
            return opAnd(opNot(self.v1.v1), opNot(self.v1.v2))
        
        # Negation Laws
        elif isinstance(self.v1, opFA):
            if isinstance(self.v1.v2, opNot):
                return opTE(self.v1.v1, self.v1.v2.v1)
        elif isinstance(self.v1, opTE):
            if isinstance(self.v1.v2, opNot):
                return opFA(self.v1.v1, self.v1.v2.v1)
        
        # Evaluate
        elif isinstance(self.v1, opVar):
            if self.v1.value is not None:
                return opVar(str(not self.v1.value))
        
        return self

class opOr(op):
    symbol = '∨'

    def simplifyStep(self):
        if isinstance(self.v1, opVar) and self.v1.value is not None:
            if self.v1.value:
                return opVar(True)
            else:
                return self.v2
        if isinstance(self.v2, opVar) and self.v2.value is not None:
            if self.v2.value:
                return opVar(True)
            else:
                return self.v1
        if self.v1.recursiveEquals(self.v2):
            return self.v1

        return self
class opAnd(op):
    symbol = '∧'

    def simplifyStep(self):
        if isinstance(self.v1, opVar) and self.v1.value is not None:
            if self.v1.value:
                return self.v2
            else:
                return opVar(False)
        if isinstance(self.v2, opVar) and self.v2.value is not None:
            if self.v2.value:
                return self.v1
            else:
                return opVar(False)
        if self.v1.recursiveEquals(self.v2):
            return self.v1

        return self
class opXor(op):
    symbol = '⊻'

    def simplifyStep(self):
        return opAnd(opNot(opAnd(self.v1, self.v2)), opOr(self.v1, self.v2))
class opFA(op):
    symbol = '∀'
    symmetric = False
class opTE(op):
    symbol = '∃'
    symmetric = False
class opIf(op):
    symbol = '⇒'
    symmetric = False
    def simplifyStep(self):
        return opOr(opNot(self.v1), self.v2)
class opImp(op):
    symbol = '⇔'

    def simplifyStep(self):
        if self.v1.recursiveEquals(self.v2):
            return opVar(True)
        return opAnd(opIf(self.v1, self.v2), opIf(self.v2, self.v1)).simplify()

class opVar(op):
    def __init__(self, name, value=None):
        if name == False or name == True:
            opVar.__init__(self, str(name), value)
        else:
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

    def recursiveEquals(self, other):
        if not isinstance(other, opVar):
            return False
        return self.name == other.name

    def simplify(self):
        return self

    def evaluate(self, globs=None):
        if globs is None:
            return self if self.value is None else opVar(self.value)

        assert isinstance(globs, dict)
        if self.name in globs.keys():
            return opVar(globs[self.name])
        elif self.value is not None:
            return opVar(self.value)
        else:
            return self

    def variables(self):
        return {self.name}

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


def ASTfromRPN(tokens):
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

def ASTfromStr(string):
    return ASTfromRPN(shuntingYard(tokenize(string)))


def truthTable(AST):
    assert isinstance(AST, op)
    v = list(AST.variables())
    n = len(v)
    o = []
    for i in range(2 ** n):
        values = [bool(int(char)) for char in '0' * (n - len(bin(i)[2:])) + bin(i)[2:]]
        globs = {v[j]: values[j] for j in range(n)}
        o.append((values, AST.evaluate(globs)))
    return o

def displayTruthTable(AST):
    def padSpaces(string, length):
        return string + (' ' * (length - len(string)))
    tt = truthTable(AST)
    o = ''
    variables = AST.variables()
    maxLen = max(5, max(len(v) for v in variables))
    for v in variables:
        o += padSpaces(v, maxLen) + ' | '
    o += 'output'
    o += '\n' + ('-' * len(o)) + '\n'
    for i, values in enumerate(tt):
        for v in values[0]:
            o += padSpaces(str(v), maxLen) + ' | '
        o += str(values[1]) + '\n'
    print(o)



if __name__ == '__main__':
    test_L = '¬(¬p∧¬q)⇔¬¬(p∨q)'
    test_AST = ASTfromStr(test_L)
    print('original', test_AST)
    print('simplified', test_AST.simplify())

    test_L = 'p⇔q'
    test_AST = ASTfromStr(test_L)
    print('original', test_AST)
    print('simplified', test_AST.simplify())

    test_L = '(¬p∧¬q)∨¬(p∨q)'
    test_AST = ASTfromStr(test_L)
    print('original', test_AST)
    print('simplified', test_AST.simplify())
