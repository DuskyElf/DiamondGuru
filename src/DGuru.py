### DIGITS ###
DIGITS = '0123456789'

### ERRORS ###
class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
    
    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fname}, line {self.pos_start.linumber + 1}'
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

### POSITION ###
class Position:
    def __init__(self, index, linumber, conumber, fname, ftext):
        self.index = index
        self.linumber = linumber
        self.conumber = conumber
        self.fname = fname
        self.ftext = ftext
    
    def increment(self, current_char):
        self.index += 1
        self.conumber += 1
        
        if current_char == '\n':
            self.linumber += 1
            self.conumber = 0
        
        return self

    def copy(self):
        return Position(self.index, self.linumber, self.conumber, self.fname, self.ftext)

### TOKENS ###

TT_INT      = 'INT'
TT_FLOAT    = 'FLOAT'
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'

class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
    
    def __repr__(self):
        if self.value: return f"{self.type}:{self.value}"
        return f'{self.type}'
    
### LEXER ###

class Lexer:
    def __init__(self, fname, text):
        self.text = text
        self.pos = Position(-1, 0, -1, fname, text)
        self.current_char = None
        self.increment()
        
    def increment(self):
        self.pos.increment(self.current_char)
        self.current_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None
    
    def make_tokens(self):
        tokens = []
        
        while self.current_char != None:
            if self.current_char in ' \t':
                self.increment()
            elif self.current_char in DIGITS:
                tokens.append(self.define_number())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.increment()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.increment()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.increment()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.increment()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.increment()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.increment()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.increment()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")
        
        return tokens, None
    
    def define_number(self):
        num_str = ''
        period_count = 0
        
        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if period_count == 1: break
                period_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.increment()
            
        if period_count == 0:
            return Token(TT_INT, int(num_str))
        else:
            return Token(TT_FLOAT, float(num_str))

### RUN ###
def run(fname, text):
    lexer = Lexer(fname, text)
    tokens, error = lexer.make_tokens()
    
    return tokens, error
