operatorTokens = '¬!∧&∨|⊻^∀∃⇒⇔='  # In order of precedence
whitespaceTokens = ' \n'
breakOnChars = operatorTokens + whitespaceTokens + '()[]{}'
operatorPrecedence = {opSymbol: i for i, opSymbol in enumerate(operatorTokens)} 


print('here are some helpful charachters for building logical statements: \n ' + operatorTokens)
print('tesing')

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
        print(tok, queue, stack)
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


test_L = '¬p∧¬q⇔¬(p∨q)'
test_tokens = tokenize(test_L)
print(test_tokens)

test_L = ' ¬p  ∧\n¬ q        ⇔¬ (                          p \n\n\n\n∨q)'
test_tokens = tokenize(test_L)
print(test_tokens)

print(shuntingYard(test_tokens))

