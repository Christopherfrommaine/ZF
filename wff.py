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
    
    def visualEquals(self, other):
        return str(self) == str(other)
    def recursiveEquals(self, other):
        if self.arity != other.arity:
            return opVar(False)
        
        match self.arity:
            case 1:
                return self.v1.recursiveEquals(other.v1)
            case 2:
                return self.v1.recursiveEquals(other.v1) and self.v2.recursiveEquals(other.v2)
            case _:
                pass

    def simplify(self):
        if hasattr(self, "simplifyStep"):
            return op.simplify(self.simplifyStep())

        match self.arity:
            case 1:
                return type(self)(self.v1.simplify())
            case 2:
                return type(self)(self.v1.simplify(), self.v2.simplify())
            case _:
                return self

class opNot(op):
    def __init__(self, val1):
        op.__init__(self, val1, None)
    
    symbol = '¬'
    arity = 1

    def simplifyStep(self):
        # Double Negation
        if isinstance(self.v1, opNot):
            return self.v1.v1
        
        # DeMorgan's laws
        elif isinstance(self.v1, opAnd):
            if isinstance(self.v1.v1, opNot) and isinstance(self.v1.v2, opNot):
                return opOr(self.v1.v1.v1, self.v1.v2.v1)
        elif isinstance(self.v1, opOr):
            if isinstance(self.v1.v1, opNot) and isinstance(self.v1.v2, opNot):
                return opAnd(self.v1.v1.v1, self.v1.v2.v1)
        
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

        return self
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

    def simplifyStep(self):
        if self.v1.recursiveEquals(self.v2):
            return opVar(True)
        return self

class opVar(op):
    def __init__(self, name, value=None):
        if name == False or name == True:
            opVar.__init__(self, str(name), value)

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

if __name__ == '__main__':
    test_L = ' ¬(¬p  ∧\n¬ q )       ⇔¬     ¬ (                          p \n\n\n\n∨q)'
    test_tokens = tokenize(test_L)
    test_RP = shuntingYard(test_tokens)
    test_AST = ASTfromRP(test_RP)
    print(test_AST)
    print(test_AST.simplify())
