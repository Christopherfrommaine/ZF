operatorTokens = '⚠¬!∧&∨|⊻^∀∃⇒⇔='  # In order of precedence, must keep the error symbol at the beginning, as it is the symbol for variables, which have precedence over everything else.
whitespaceTokens = ' \n'
breakOnChars = operatorTokens + whitespaceTokens + '()[]{}'
operatorPrecedence = {opSymbol: i for i, opSymbol in enumerate(operatorTokens)}

if __name__ == '__main__':
    print('here are some helpful charachters for building logical statements: \n ' + operatorTokens + '\n')


from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed

class op:
    def __init__(self, val1, val2):
        self.v1 = val1
        self.v2 = val2

    arity = 2
    symbol = '⚠'
    symmetric = True

    def __repr__(self):
        inParens = lambda v: f'({v})' if operatorPrecedence[type(self).symbol] < operatorPrecedence[
            type(v).symbol] else f'{v}'
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

    def __eq__(self, other):
        return self.ttEquals(other)

    def __hash__(self):
        return hash(str(self))

    def visualEquals(self, other):
        return str(self) == str(other)

    def recursiveEquals(self, other):
        if self.arity != other.arity:
            return False

        if self.arity == 1:
            return self.v1.recursiveEquals(other.v1)

        if self.symmetric:
            return (self.v1.recursiveEquals(other.v1) and self.v2.recursiveEquals(other.v2)) or (
                        self.v1.recursiveEquals(other.v2) and self.v2.recursiveEquals(other.v1))
        else:
            return self.v1.recursiveEquals(other.v1) and self.v2.recursiveEquals(other.v2)

    def ttEquals(self, other):
        return all([element[1] for element in truthTable(opImp(self, other))])

    def variables(self):
        if self.arity == 1:
            return self.v1.variables()
        return set.union(self.v1.variables(), self.v2.variables())

    def opNumber(self):
        return 1 + self.v1.opNumber() + self.v2.opNumber()

    def evaluate(self, globs=None):
        return opVar('invalid: no op type')

    # Simplification
    def simplifyStep(self, v1, v2):
        return [type(self)(v1, v2)]

    def expandSimplification(self, maxDepth, maxBranches):
        pv1 = self.v1.expandSimplification(maxDepth - 1, maxBranches)
        pv2 = self.v2.expandSimplification(maxDepth - 1, maxBranches)

        simplifications = set()
        for v1 in pv1:
            for v2 in pv2:
                simplifications = simplifications.union(set(self.simplifyStep(v1, v2)))
        #
        # if maxDepth >= 1:
        #     print(maxDepth, len(simplifications))

        return sorted(simplifications, key=lambda x: x.opNumber())[:maxBranches]

    # def simplify(self, maxDepth=3, maxBranches=1000, maxTopLevelBranches=50):
    #     o = {self}
    #     for i in range(maxDepth):
    #         print(f'i: {i}, maxDepth: {maxDepth}, len(o): {len(o)}, _______')
    #         oldo = o.copy()
    #         for simpi, simp in enumerate(oldo):
    #             print(f'{simpi} / {len(oldo)} | new simp! --------------------')
    #             o = set.union(o, simp.expandSimplification(maxDepth, maxBranches))
    #         o = set(sorted(o, key=lambda x: x.opNumber())[:maxTopLevelBranches])
    #         if oldo == o:
    #             print('no more possible simplifications')
    #             break
    #     o = sorted(o, key=lambda x: x.opNumber())
    #     return o[0]


    def simplify(self, maxDepth=3, maxBranches=1000, maxTopLevelBranches=50):
        o = {self}
        for i in range(maxDepth):
            print(f'i: {i}, maxDepth: {maxDepth}, len(o): {len(o)}, _______')
            oldo = o.copy()

            # Use ProcessPoolExecutor to parallelize the expansion
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(simp.expandSimplification, maxDepth, maxBranches): simp for simp in oldo}
                for future in as_completed(futures):
                    simp = futures[future]
                    try:
                        result = future.result()
                        o = set.union(o, result)
                    except Exception as e:
                        print(f'Exception: {e}')

            o = set(sorted(o, key=lambda x: x.opNumber())[:maxTopLevelBranches])
            if oldo == o:
                print('no more possible simplifications')
                break

        o = sorted(o, key=lambda x: x.opNumber())
        return o[0]


class opNot(op):
    def __init__(self, val1):
        op.__init__(self, val1, None)

    symbol = '¬'
    arity = 1
    def evaluate(self, globs=None):
        return not self.v1.numEval__(globs)

    def opNumber(self):
        return 1 + self.v1.opNumber()

    def expandSimplification(self, maxDepth, maxBranches):
        pv1 = self.v1.expandSimplification(maxDepth - 1, maxBranches)

        simplifications = []
        for v1 in pv1:
            simplifications += self.simplifyStep(v1, None)

        if len(simplifications) >= maxBranches:
            print('original: ', simplifications)
            simplifications = sorted(simplifications, key=lambda x: x.opNumber())[:maxBranches]
            print('sorted: ', simplifications)

        return simplifications

    def simplifyStep(self, v1, v2):
        o = [type(self)(v1)]

        # Evaluation
        if isinstance(v1, opVar) and v1.value is not None:
            o.append(opVar(not v1.value))

        # Double Negation
        elif isinstance(v1, opNot):
            o.append(v1.v1)

        # DeMorgan's laws
        elif isinstance(v1, opAnd):
            o.append(opOr(opNot(v1.v1), opNot(v1.v2)))
        elif isinstance(v1, opOr):
            o.append(opAnd(opNot(v1.v1), opNot(v1.v2)))

        return o


class opOr(op):
    symbol = '∨'
    def evaluate(self, globs=None):
        return self.v1.numEval__(globs) or self.v2.numEval__(globs)

    def simplifyStep(self, v1, v2):
        o = [type(self)(v1, v2)]

        if isinstance(v1, opVar) and v1.value is not None:
            o.append(opVar(True) if v1.value else v2)
        if isinstance(v2, opVar) and v2.value is not None:
            o.append(opVar(True) if v2.value else v1)

        if v1 == v2:
            o.append(v1)

        return o


class opAnd(op):
    symbol = '∧'

    def evaluate(self, globs=None):
        return self.v1.numEval__(globs) and self.v2.numEval__(globs)

    def simplifyStep(self, v1, v2):
        o = [type(self)(v1, v2)]

        if isinstance(v1, opVar) and v1.value is not None:
            o.append(v2 if v1.value else opVar(False))
        if isinstance(v2, opVar) and v2.value is not None:
            o.append(v1 if v2.value else opVar(False))

        if v1 == v2:
            o.append(v1)

        return o


class opXor(op):
    symbol = '⊻'

    def evaluate(self, globs=None):
        v1, v2 = self.v1.numEval__(globs), self.v2.numEval__(globs)
        return (v1 or v2) and not (v1 and v2)

    def simplifyStep(self, v1, v2):
        return [type(self)(v1, v2), opAnd(opNot(opAnd(self.v1, self.v2)), opOr(self.v1, self.v2))]

class opIf(op):
    symbol = '⇒'
    symmetric = False

    def evaluate(self, globs=None):
        v1, v2 = self.v1.numEval__(globs), self.v2.numEval__(globs)
        return not v1 or v2

    def simplifyStep(self, v1, v2):
        return [type(self)(v1, v2), opOr(opNot(v1), v2)]

class opImp(op):
    symbol = '⇔'
    asFunc = lambda x, y: x == y

    def evaluate(self, globs=None):
        v1, v2 = self.v1.numEval__(globs), self.v2.numEval__(globs)
        return not (v1 or v2) or (v1 and v2)

    def simplifyStep(self, v1, v2):
        if v1.ttEquals(v2):
            return [opVar(True)]
        return [type(self)(v1, v2), opAnd(opIf(v1, v2), opIf(v2, v1))]


class opVar(op):
    def __init__(self, name, value=None):
        if name is False or name is True:
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

    def evaluate(self, globs=None):
        if globs is None:
            return self if self.value is None else self.value

        assert isinstance(globs, dict)
        if self.name in globs.keys():
            return globs[self.name]
        elif self.value is not None:
            return self.value
        else:
            return self

    def variables(self):
        return {self.name}

    def opNumber(self):
        return 1

    def simplifyStep(self, v1, v2):
        return [self]

    def expandSimplification(self, maxDepth, maxBranches):
        return [self]

    def simplify(self, *args, **kwargs):
        return self


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
    # print(ASTfromStr('p') == ASTfromStr('q'))
    #
    # test_L = '¬(¬p∧¬q)⇔¬¬(p∨q)'
    # test_AST = ASTfromStr(test_L)
    # print(ASTfromStr('¬(¬p∧¬q)') == ASTfromStr('¬¬(p∨q)'))
    #
    # print('original', test_AST)
    # displayTruthTable(test_AST)
    #
    #
    # print('simplified', test_AST.simplify())
    #
    # test_L = 'p⇔q'
    # test_AST = ASTfromStr(test_L)
    # print('original', test_AST)
    # print('simplified', test_AST.simplify())
    #
    # test_L = '(¬p∧¬q)∨¬(p∨q)'
    # test_AST = ASTfromStr(test_L)
    # print('original', test_AST)
    # print('simplified', test_AST.simplify())

    finalBoss_L = '(¬p∨q)∧(¬r∧s)∨(t∧¬p)∨(q∧r∧¬s)∨(¬t∧(p⊻q))∧(¬r⇒(s⇔t))∨((p∧¬q)∨(r∧s)∧¬(t∨¬p))∧(¬q⊻(r∨¬s))∧(p∨¬r)∨(¬s∧t)∧(p⇔¬q)∨(r⇒(¬s∨t))∧(¬p∧q)∨(r⊻¬s)∧(t∨¬p∧q)∨((r∧¬s)∨t)∧(¬p⊻q)∨(r∧¬s∧t)∨(¬p⇒(q∧r))∧(s∨¬t)'
    finalBoss_AST = ASTfromStr(finalBoss_L)

    finalBossOut_L = ' (¬p∨q)∧¬r∧s∨t∧¬p∨q∧r∧¬s∨¬t∧(p⊻q)∧(r∨(s⇔t))∨(p∧¬q∨r∧s∧¬t∧p)∧(¬q⊻r∨¬s)∧(p∨¬r)∨¬s∧t∧(p⇔¬q)∨(r⇒¬s∨t)∧¬p∧q∨(r⊻¬s)∧(t∨¬p∧q)∨(r∧¬s∨t)∧(¬p⊻q)∨r∧¬s∧t∨(p∨q∧r)∧(s∨¬t)'
    finalBossOut_AST = ASTfromStr(finalBossOut_L)

    displayTruthTable(finalBoss_AST)
    displayTruthTable(finalBossOut_AST)
    print(finalBossOut_AST == finalBoss_AST)

    print(finalBoss_AST.simplify(5, 1000, 50))
