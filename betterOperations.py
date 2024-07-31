operatorTokens = '⚠¬!∧&∨|⊻^∀∃⇒⇔='  # In order of precedence, must keep the error symbol at the beginning, as it is the symbol for variables, which have precedence over everything else.
whitespaceTokens = ' \n'
breakOnChars = operatorTokens + whitespaceTokens + '()[]{}'
operatorPrecedence = {opSymbol: i for i, opSymbol in enumerate(operatorTokens)}

if __name__ == '__main__':
    print('here are some helpful charachters for building logical statements: \n ' + operatorTokens + '\n')


from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed

class op:
    def __init__(self, *args):
        assert len(args) == self.arity
        self.v = args

    # Default for most operation types
    arity = 2
    symbol = '⚠'
    symmetric = True


    def __repr__(self):
        inParens = lambda v: f'({v})' if operatorPrecedence[self.symbol] < operatorPrecedence[v.symbol] else f'{v}'
        if self.arity == 1:
            return self.symbol + inParens(self.v[0])
        elif self.arity == 2:
            return inParens(self.v[0]) + self.symbol + inParens(self.v[1])
        else:
            return self.symbol + str(tuple(self.v))

    def __invert__(self):
        return opNot(self)
    def __or__(self, other):
        return opOr(self, other)
    def __and__(self, other):
        return opAnd(self, other)
    def __xor__(self, other):
        return opXor(self, other)
    def __ror__(self, other):
        return self.__or__(other)
    def __rand__(self, other):
        return self.__and__(other)
    def __rxor__(self, other):
        return self.__xor__(other)
    def __eq__(self, other):
        return self.ttEquals(other)
    def __hash__(self):
        return hash(str(self))


    def visualEquals(self, other):
        return str(self) == str(other)
    def recursiveEquals(self, other, overrideSymmetry=False):
        if self.arity != other.arity:
            return False
        if self.symmetric and not overrideSymmetry:
            return self.recursiveEquals(other, overrideSymmetry=True) or other.recursiveEquals(self, overrideSymmetry=True)
        else:
            return all(self.v[i].recursiveEquals(other.v[i]) for i in range(self.arity))
    def ttEquals(self, other):
        return all([element[1] for element in truthTable(opImp(self, other))])

    def variables(self):
        return set(sum([list(vi.variables()) for vi in self.v], start=[]))
    def opNumber(self):
        return 1 + sum(vi.opNumber() for vi in self.v)
    def maxDepth(self):
        return 1 + max([0] + [vi.depth() for vi in self.v])

    def eval(self, globs=None):
        o = self.numEval__(globs)
        if isinstance(o, op):
            def cleanupNumbersInOp(opi):
                for i, vi in enumerate(opi.v):
                    if isinstance(vi, op):
                        cleanupNumbersInOp(vi)
                    else:
                        opi.v[i] = opVar(str(vi), bool(vi & 1))
            cleanupNumbersInOp(o)
            return o
        return bool(o & 1)

    def numEval__(self, globs):
        return opVar('invalid: no op type')

    # Simplification
    def simplifyStep(self, v):
        return [type(self)(*v)]

    def expandSimplification(self, maxBranches):
        def simplifyByArgument(remainingp, args):
            if not len(remainingp):
                return set(self.simplifyStep(args))
            o = set()
            for vi in remainingp[0]:
                o = o.union(simplifyByArgument(remainingp[1:], args=args + [vi]))
            return o

        try:
            return sorted(
                simplifyByArgument(
                    [vi.expandSimplification(maxBranches) for vi in self.v], []),
                key=lambda x: x.opNumber())[:maxBranches]
        except:
            a = [vi.expandSimplification(maxBranches) for vi in self.v]
            b = simplifyByArgument(a, [])
            c = sorted(b, key=lambda x: x.opNumber())[:maxBranches]
            return c



    def simplify(self, maxRepititions=3, maxBranches=1000, maxTopLevelBranches=50):
        o = {self}
        for _ in range(maxRepititions):
            oldo = o.copy()
            for simp in oldo:
                o = o.union(simp.expandSimplification(maxBranches))
            o = set(sorted(o, key=lambda x: x.opNumber())[:maxTopLevelBranches])
            if oldo == o:
                break
        return sorted(o, key=lambda x: x.opNumber())[0]


class opNot(op):
    symbol = '¬'
    arity = 1

    def numEval__(self, globs):
        return ~self.v[0].numEval__(globs)

    def simplifyStep(self, v):
        o = [type(self)(*v), type(self)(*v).eval()]

        # Double Negation
        if isinstance(v[0], opNot):
            o.append(v[0].v[0])

        # DeMorgan's laws
        elif isinstance(v[0], opAnd):
            o.append(opOr(opNot(v[0].v[0]), opNot(v[0].v[1])))
        elif isinstance(v[0], opOr):
            o.append(opAnd(opNot(v[0].v[0]), opNot(v[0].v[1])))

        return o


class opOr(op):
    symbol = '∨'

    def numEval__(self, globs):
        return self.v[0].numEval__(globs) | self.v[1].numEval__(globs)

    def simplifyStep(self, v):
        # Original, Evaluated, Commutativiy
        o = [type(self)(*v), type(self)(*v).eval(), type(self)(*reversed(v))]

        # Special Evaluation
        if isinstance(v[0], opVar) and v[0].value is not None:
            o.append(opVar(True) if v[0].value else v[1])

        # Associativity
        if isinstance(v[0], opOr):
            o.append(opOr(v[0].v[0], opOr(v[0].v[1], v[1])))

        # Equal Arguments
        if v[0] == v[1]:
            o.append(v[0])

        return o


class opAnd(op):
    symbol = '∧'

    def numEval__(self, globs):
        return self.v[0].numEval__(globs) & self.v[1].numEval__(globs)

    def simplifyStep(self, v):
        # Original, Evaluated, Commutativiy
        o = [type(self)(*v), type(self)(*v).eval(), type(self)(*reversed(v))]

        # Special Evaluation
        if isinstance(v[0], opVar) and v[0].value is not None:
            o.append(v[1] if v[0].value else opVar(False))

        # Associativity
        if isinstance(v[0], opAnd):
            o.append(opAnd(v[0].v[0], opAnd(v[0].v[1], v[1])))

        # Equal Arguments
        if v[0] == v[1]:
            o.append(v[0])

        return o


class opXor(op):
    symbol = '⊻'

    def numEval__(self, globs):
        return self.v[0].numEval__(globs) ^ self.v[1].numEval__(globs)

    def simplifyStep(self, v):
        v1, v2, = v[:2]
        return [type(self)(v1, v2), opAnd(opNot(opAnd(v1, v2)), opOr(v1, v2))]

class opIf(op):
    symbol = '⇒'
    symmetric = False

    def numEval__(self, globs):
        return ~self.v[0].numEval__(globs) | self.v[1].numEval__(globs)

    def simplifyStep(self, v):
        v1, v2, = v[:2]
        return [type(self)(v1, v2), opOr(opNot(v1), v2)]

class opImp(op):
    symbol = '⇔'

    def numEval__(self, globs):
        v1, v2 = self.v[0].numEval__(globs), self.v[1].numEval__(globs)
        return not (v1 or v2) or (v1 and v2)

    def simplifyStep(self, v):
        v1, v2, = v[:2]
        if v1.ttEquals(v2):
            return [opVar(True)]
        return [type(self)(v1, v2), opAnd(opIf(v1, v2), opIf(v2, v1))]


class opVar(op):
    def __init__(self, name, value=None):
        if name is False or name is True:
            opVar.__init__(self, str(name), value)
        else:
            op.__init__(self)
            self.name = name
            self.value = value

            if self.name == 'True':
                self.value = True
            if self.name == 'False':
                self.value = False

    arity = 0
    symbol = '⚠'

    def __repr__(self):
        return self.name

    def recursiveEquals(self, other, overrideSimplify=False):
        if not isinstance(other, opVar):
            return False
        return self.name == other.name

    def numEval__(self, globs):
        globs = {} if globs is None else globs

        value = globs[self.name] if self.name in globs.keys() else self.value

        if value is None:
            return self
        else:
            return int(value)

    def variables(self):
        return {self.name}

    def simplifyStep(self, v):
        return [self]


# '∀': opFA, '∃': opTE
opFromTok = {'¬': opNot, '!': opNot, '∧': opAnd, '&': opAnd, '∨': opOr, '|': opOr, '⊻': opXor, '^': opXor, '⇒': opIf, '⇔': opImp, '=': opImp}


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
        o.append((values, AST.eval(globs)))
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

    print(ASTfromStr('p') == ASTfromStr('p'))
    print(ASTfromStr('p') == ASTfromStr('q'))

    test_L = '¬(¬p∧¬q)⇔¬¬(p∨q)'
    test_AST = ASTfromStr(test_L)
    print(ASTfromStr('¬(¬p∧¬q)') == ASTfromStr('¬¬(p∨q)'))

    print('original', test_AST)
    displayTruthTable(test_AST)


    print('simplified', test_AST.simplify())

    test_L = 'p⇔q'
    test_AST = ASTfromStr(test_L)
    print('original', test_AST)
    print('simplified', test_AST.simplify())

    test_L = '(¬p∧¬q)∨¬(p∨q)'
    test_AST = ASTfromStr(test_L)
    print('original', test_AST)
    print('simplified', test_AST.simplify(maxBranches=5))
    #
    # finalBoss_L = '(¬p∨q)∧(¬r∧s)∨(t∧¬p)∨(q∧r∧¬s)∨(¬t∧(p⊻q))∧(¬r⇒(s⇔t))∨((p∧¬q)∨(r∧s)∧¬(t∨¬p))∧(¬q⊻(r∨¬s))∧(p∨¬r)∨(¬s∧t)∧(p⇔¬q)∨(r⇒(¬s∨t))∧(¬p∧q)∨(r⊻¬s)∧(t∨¬p∧q)∨((r∧¬s)∨t)∧(¬p⊻q)∨(r∧¬s∧t)∨(¬p⇒(q∧r))∧(s∨¬t)'
    # finalBoss_AST = ASTfromStr(finalBoss_L)
    #
    # finalBossOut_L = ' (¬p∨q)∧¬r∧s∨t∧¬p∨q∧r∧¬s∨¬t∧(p⊻q)∧(r∨(s⇔t))∨(p∧¬q∨r∧s∧¬t∧p)∧(¬q⊻r∨¬s)∧(p∨¬r)∨¬s∧t∧(p⇔¬q)∨(r⇒¬s∨t)∧¬p∧q∨(r⊻¬s)∧(t∨¬p∧q)∨(r∧¬s∨t)∧(¬p⊻q)∨r∧¬s∧t∨(p∨q∧r)∧(s∨¬t)'
    # finalBossOut_AST = ASTfromStr(finalBossOut_L)
    #
    # displayTruthTable(finalBoss_AST)
    # displayTruthTable(finalBossOut_AST)
    # print(finalBossOut_AST == finalBoss_AST)
    #
    # print(finalBoss_AST.simplify(5, 1000, 50))
