print('here are some helpful charachters for building logical statements: \n ¬∧∨∀∃⇔∅∈')

class wff:
    def __init__(self, stringRepr='T'):
        self.str = stringRepr

        self.variables = []
        self.ast = None