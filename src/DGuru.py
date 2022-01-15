import sys
import subprocess
import string

### CONSTANTS ###
DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS =  LETTERS + DIGITS

### Super String ###
def super_string(text, pos_start, pos_end):
    result = ''
    
    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.index), 0)
    idx_end = text.rfind('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
    
    # Gnerate lines
    line_count = pos_end.linumber - pos_start.linumber + 1
    for i in range(line_count):
        line = text[idx_start:idx_end]
        col_start = pos_start.conumber if i == 0 else 0
        col_end = pos_end.conumber if i == line_count - 1 else len(line) - 1
        
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)
        
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
    
    return result.replace('\t', '')

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
        result += '\n\n' + super_string(self.pos_start.ftext, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "IllegalCharError", details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

### POSITION ###
class Position:
    def __init__(self, index, linumber, conumber, fname, ftext):
        self.index = index
        self.linumber = linumber
        self.conumber = conumber
        self.fname = fname
        self.ftext = ftext
    
    def increment(self, current_char=None):
        self.index += 1
        self.conumber += 1
        
        if current_char == '\n':
            self.linumber += 1
            self.conumber = 0
        
        return self

    def copy(self):
        return Position(self.index, self.linumber, self.conumber, self.fname, self.ftext)

### TOKENS ###

TT_INT          = 'INT'
TT_FLOAT        = 'FLOAT'
TT_IDENTIFIER   = 'IDENTIFIER'
TT_KEYWORD      = 'KEYWORD'
TT_PLUS         = 'PLUS'
TT_MINUS        = 'MINUS'
TT_MUL          = 'MUL'
TT_DIV          = 'DIV'
TT_POW          = 'POW'
TT_EQ           = 'EQ'
TT_LPAREN       = 'LPAREN'
TT_RPAREN       = 'RPAREN'
TT_NEWLINE      = 'NEWLINE'
TT_EOF          = 'EOF'

KEYWORDS = []

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.increment()
        
        if pos_end:
            self.pos_end = pos_end.copy()
    
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
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.increment()
            elif self.current_char in DIGITS:
                tokens.append(self.define_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.increment()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.increment()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.increment()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.increment()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.increment()
            elif self.current_char == '=':
                tokens.append(Token(TT_EQ, pos_start=self.pos))
                self.increment()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.increment()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.increment()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.increment()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")
        
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None
    
    def define_number(self):
        num_str = ''
        period_count = 0
        pos_start = self.pos.copy()
        
        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if period_count == 1: break
                period_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.increment()
            
        if period_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)
    
    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()
        
        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.increment()
        
        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

### NODES ###
class NumberNode:
    def __init__(self, token):
        self.token = token
        
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end
    
    def __repr__(self):
        return f"{self.token.value}"

class IdentifierNode:
    def __init__(self, id_name_token):
        self.id_name_token = id_name_token
        
        self.pos_start = self.id_name_token.pos_start
        self.pos_end = self.id_name_token.pos_end
    
    def __repr__(self):
        return f"{self.id_name_token.value}"

class IdAssignNode:
    def __init__(self, id_name_token, value_node):
        self.id_name_token = id_name_token
        self.value_node = value_node
        
        self.pos_start = self.id_name_token.pos_start
        self.pos_end = self.value_node.pos_end
    
    def __repr__(self):
        return f"IdentifierAssign({self.id_name_token.value}, {self.value_node})"

class BinOpNode:
    def __init__(self, left_node, op_token, right_node):
        self.left_node = left_node
        self.op_token = op_token
        self.right_node = right_node
        
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end
    
    def __repr__(self):
        return f"({self.left_node}, {self.op_token}, {self.right_node})"

class UnaryOpNode:
    def __init__(self, op_token, node):
        self.op_token = op_token
        self.node = node
        
        self.pos_start = self.op_token.pos_start
        self.pos_end = self.node.pos_end
    
    def __repr__(self):
        return f"({self.op_token}, {self.node})"

class StatementNode:
    def __init__(self, statement_node):
        self.statement_node = statement_node
    
    def __repr__(self):
        return f'{self.statement_node}'

class CodeNode:
    def __init__(self, statements):
        self.statements = statements
    
    def __repr__(self):
        return f'{self.statements}'

### PARSE RESULT
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
    
    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        
        return res
    
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        self.error = error
        return self

### PARSER ###
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = -1
        self.increment()
    
    def increment(self):
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
        return self.current_token
    
    def parser(self):
        res = self.code()
        if not res.error and self.current_token.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '+', '-', '*' or '/'"
            ))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.current_token
        
        if tok.type in (TT_INT, TT_FLOAT):
            res.register(self.increment())
            return res.success(NumberNode(tok))
        
        elif tok.type == TT_IDENTIFIER:
            res.register(self.increment())
            return res.success(IdentifierNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.increment())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_token.type == TT_RPAREN:
                res.register(self.increment())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected ')'"
                ))
        
        return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected int, float, '+', '-' or '('"
                ))

    def power(self):
        return self.bin_op(self.atom, (TT_POW,), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.current_token
        
        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.increment())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))
        
        return self.power()
    
    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))
    
    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))
    
    def identifier(self):
        res = ParseResult()
        identifier_name = self.current_token
        res.register(self.increment())
        
        if self.current_token.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '='"
            )) # This error might never occur
        
        res.register(self.increment())
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(IdAssignNode(identifier_name, expr))
    
    def statement(self):
        res = ParseResult()
        if self.current_token.type == TT_IDENTIFIER and self.tokens[self.token_index + 1].type == TT_EQ:
            statement = res.register(self.identifier())
        else:
            statement = res.register(self.expr())
        
        if res.error: return res
        return res.success(StatementNode(statement))
    
    def code(self):
        res = ParseResult()
        statements = []
        while self.current_token.type != TT_EOF:
            statements.append(res.register(self.statement()))
            if res.error: return res
            temp = False
            while self.current_token.type == TT_NEWLINE:
                res.register(self.increment())
                temp = True
            if not temp:
                if self.current_token.type == TT_EOF:
                    return res.success(CodeNode(statements))
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '\\n' or ';'"
                    ))
        return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected statement, empty files are not compilable"
            ))
    
    #######################################################
    
    def bin_op(self, func_a, ops, func_b=None):
        if func_b is None: func_b = func_a
        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_token.type in ops:
            op_token = self.current_token
            res.register(self.increment())
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_token, right)
            
        return res.success(left)

### Writer ###
class Writer:
    def __init__(self):
        self.core_code = ''
        self.libraries = {"#include<stdlib.h>"}
        self.identifiers = {}
    
    def write_code(self, code):
        self.core_code += code + ';\n'
    
    def result(self):
        self.core_code = f"int main(){{\n{self.core_code}\nreturn 0;\n}}"
        self.core_code = f"void* identifiers[{len(self.identifiers)}];\n{self.core_code}"
        for library in self.libraries:
            self.core_code = f'{library}\n{self.core_code}'
        return self.core_code

### Analizer ###
class Analizer:    
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_node)
        return method(node)
    
    def no_visit_node(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined.')
    
    def visit_NumberNode(self, node):
        return node.token.value, type(node.token.value)
    
    def visit_IdentifierNode(self, node):
        def map_type(type_):
            if type_ == 'int': return int
            if type_ == 'double': return float
        
        if node.id_name_token.value not in self.writer.identifiers.keys(): return
        d = self.writer.identifiers[node.id_name_token.value]
        return f"(*({d[1]}*)identifiers[{d[0]}])", map_type(d[1])
    
    def visit_IdAssignNode(self, node):
        def map_type(type_):
            if type_ is int: return 'int'
            if type_ is float: return 'double'
        
        value_node = self.visit(node.value_node)
        name = node.id_name_token.value
        id_list = self.writer.identifiers
        
        if name in id_list.keys():
            if map_type(value_node[1]) == id_list[name][1]:
                return f"*({id_list[name][1]}*)identifiers[{id_list[name][0]}] = {value_node[0]}", value_node[1]
            
            self.writer.write_code(f"identifiers[{id_list[name][0]}] = realloc(identifiers[{id_list[name][0]}], sizeof({map_type(value_node[1])}))")
            id_list[name][1] = map_type(value_node[1])
            return f"*({map_type(value_node[1])}*)identifiers[{id_list[name][0]}] = {value_node[0]}", value_node[1]

        id_list[name] = [-1, map_type(value_node[1])]
        id_list[name][0] = list(id_list.keys()).index(name)
        self.writer.write_code(f"identifiers[{id_list[name][0]}] = calloc(1, sizeof({map_type(value_node[1])}))")
        return f"*({map_type(value_node[1])}*)identifiers[{id_list[name][0]}] = {value_node[0]}", value_node[1]
    
    def visit_BinOpNode(self, node):
        left_node = self.visit(node.left_node)
        right_node = self.visit(node.right_node)

        if left_node[1] is float or right_node[1] is float:
            eval_type = float
        else: eval_type = int
        
        if node.op_token.type == TT_PLUS:
            return f'({left_node[0]}+{right_node[0]})', eval_type
        if node.op_token.type == TT_MINUS:
            return f'({left_node[0]}-{right_node[0]})', eval_type
        if node.op_token.type == TT_MUL:
            return f'({left_node[0]}*{right_node[0]})', eval_type
        if node.op_token.type == TT_DIV:
            return f'({left_node[0]}/(double){right_node[0]})', float
        if node.op_token.type == TT_POW:
            self.writer.libraries.add("#include<math.h>")
            return f'(pow((double){self.visit(node.left_node)}, (double){self.visit(node.right_node)}))', float
    
    def visit_UnaryOpNode(self, node):
        value_node = self.visit(node.node)
        if node.op_token.type == TT_PLUS:
            return f'(+{value_node[0]})', value_node[1]
        if node.op_token.type == TT_MINUS:
            return f'(-{value_node[0]})', value_node[1]
    
    def visit_StatementNode(self, node):
        statement_node = self.visit(node.statement_node)
        return statement_node[0], statement_node[1]
    
    def visit_CodeNode(self, node):
        self.writer = Writer()
        for statement in node.statements:
            self.writer.write_code(self.visit(statement)[0])
        return self.writer.result()

### RUN ###
def lex(file_name, context):
    # Lexing
    lexer = Lexer(file_name, context)
    return lexer.make_tokens()

def parse(tokens):
    # Parsing
    parser = Parser(tokens)
    return parser.parser()

def analize(abstractSyntaxTree):
    # analizing
    analizer = Analizer()
    return analizer.visit(abstractSyntaxTree.node)

def run(file_name, context):
    tokens, error = lex(file_name, context)
    if error: return None, error
    
    ast = parse(tokens)
    if ast.error: return None, ast.error
    
    return analize(ast), None

def open_code(file_name):
    try:
        with open(file_name, 'r') as f:
            context = f.read()
            return context
    except FileNotFoundError as exeption:
        print(exeption)
        sys.exit()

def main():
    if len(sys.argv) < 2:
        print("""Please provide any arguments:
    compile <file_name>
    c <file_name>
    lex <file_name>
    parse <file_name>
    """)
        sys.exit()
    
    if len(sys.argv) < 3:
        print("Please also provide the file path of the script :)")
        sys.exit()
    
    cmd = sys.argv[1].lower()
    file_name = sys.argv[2]
    context = open_code(file_name)
    
    if cmd == 'compile':
        result, error = run(file_name, context)
        if error: print(error.as_string())
        else:
            xfile = file_name.split('.')[0]
            with open(xfile+'.c', 'w') as f:
                f.write(result)
            try:
                subprocess.Popen(f"gcc {xfile+'.c'} -o {xfile+'.exe'}")
            except FileNotFoundError:
                print("gcc is not installed, you need gcc compiler to use this program ;(")
    
    elif cmd == 'c':
        result, error = run(file_name, context)
        if error: print(error.as_string())
        else:
            print(result)
    
    elif cmd == 'lex':
        result, error = lex(file_name, context)
        if error: print(error.as_string())
        else:
            print(result)
    
    elif cmd == 'parse':
        tokens, error = lex(file_name, context)
        if error: print(error.as_string())
        else:
            ast = parse(tokens)
            if ast.error: print(ast.error.as_string())
            else:
                print(ast.node)
    
    else:
        print(f"""Invalid argument {cmd} :(
Please provide one of the valid arguments:
    compile <file_name>
    c <file_name>
    lex <file_name>
    parse <file_name>
    """)
        sys.exit()

if __name__ == "__main__":
    main()
