import sys
import subprocess
import string

from numpy import isin

### CONSTANTS ###
DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS =  LETTERS + DIGITS

### Super String ###
def super_string(text, pos_start, pos_end):
    result = ''
    
    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.index), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
    
    # Gnerate lines
    line_count = pos_end.linumber - pos_start.linumber + 1
    for i in range(line_count):
        line = text[idx_start:idx_end]
        col_start = pos_start.conumber if i == 0 else 0
        col_end = pos_end.conumber if i == line_count - 1 else len(line) - 1

        result += line.replace('\n', '') + '\n'
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

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'ExpectedCharError', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'InvalidSyntaxError', details)

class NameError_(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'NameError', details)

class TypeError_(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'TypeError', details)

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
TT_EE           = 'EE'
TT_NE           = 'NE'
TT_LT           = 'LT'
TT_GT           = 'GT'
TT_LTE          = 'LTE'
TT_GTE          = 'GTE'
TT_NEWLINE      = 'NEWLINE'
TT_EOF          = 'EOF'

KEYWORDS = [
    'static',
    'and',
    'or',
    'not',
    'True',
    'False',
    'if',
    'elif',
    'else',
    'end',
    'choice',
    'int',
    'float',
    'bool'
]

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
                tokens.append(self.make_newline())
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
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.increment()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.increment()
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.increment()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")
        
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None
    
    def make_newline(self):
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in ';\n':
            self.increment()
        
        return Token(TT_NEWLINE, pos_start=pos_start, pos_end=self.pos)
    
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
    
    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.increment()
        
        if self.current_char == '=':
            self.increment()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None
        
        self.increment()
        return None, ExpectedCharError(pos_start, self.pos, "Expected '=' (after '!')")
    
    def make_equals(self):
        tok_type = TT_EQ
        pos_start = self.pos.copy()
        self.increment()
        
        if self.current_char == '=':
            self.increment()
            tok_type = TT_EE
        
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
    
    def make_less_than(self):
        tok_type = TT_LT
        pos_start = self.pos.copy()
        self.increment()
        
        if self.current_char == '=':
            self.increment()
            tok_type = TT_LTE
        
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
    
    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.increment()
        
        if self.current_char == '=':
            self.increment()
            tok_type = TT_GTE
        
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

### NODES ###
class NumberNode:
    def __init__(self, token):
        self.token = token
        
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end
    
    def __repr__(self):
        return f"{self.token.value}"

class KeywordValueNode:
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
    def __init__(self, id_name_token, value_node, manual_static, manual_static_token):
        self.id_name_token = id_name_token
        self.value_node = value_node
        self.manual_static = manual_static
        self.manual_static_token = manual_static_token
        self.pos_start = self.id_name_token.pos_start
        self.pos_end = self.value_node.pos_end
    
    def __repr__(self):
        return f"IdentifierAssign({self.id_name_token.value}, {self.value_node})"

class IfNode:
    def __init__(self, if_value, elif_values, else_value, pos_start, pos_end):
        self.if_value = if_value
        self.elif_values = elif_values
        self.else_value = else_value
        
        self.pos_start = pos_start
        self.pos_end = pos_end
    
    def __repr__(self):
        return f"if({self.if_value}, {self.elif_values}, {self.else_value})"

class TypeChoiceNode:
    def __init__(self, identifier, type_, expr):
        self.identifier = identifier
        self.type = type_
        self.expr = expr
        
        self.pos_start = self.identifier.pos_start
        self.pos_end = self.expr.pos_end
    
    def __repr__(self):
        return f"TypeChoice({self.identifier}, {self.type}, {self.expr})"

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
        self.pos_start = statement_node.pos_start
        self.pos_end = statement_node.pos_end
    
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
        
        elif tok.type == TT_KEYWORD and tok.value == 'True':
            res.register(self.increment())
            return res.success(KeywordValueNode(tok))
        
        elif tok.type == TT_KEYWORD and tok.value == 'False':
            res.register(self.increment())
            return res.success(KeywordValueNode(tok))

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
                    "Expected expression or '('"
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
    
    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))
    
    def comp_expr(self):
        res = ParseResult()
        
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'not':
            op_token = self.current_token
            res.register(self.increment())
            
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_token, node))
        
        node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
        if res.error:
            return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected expression, '(' or 'not'"
                ))
        
        return res.success(node)
    
    def identifier(self):
        res = ParseResult()
        manual_static = False
        manual_static_token = None
        
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'static':
            manual_static = True
            manual_static_token = self.current_token
            res.register(self.increment())
        
        if self.current_token.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected valid identifier name"
            ))
        
        identifier_name = self.current_token
        res.register(self.increment())
        
        if self.current_token.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '='"
            ))
        
        res.register(self.increment())
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(IdAssignNode(identifier_name, expr, manual_static, manual_static_token))
    
    def expr(self):
        res = ParseResult()
        if (self.current_token.type == TT_IDENTIFIER and self.tokens[self.token_index + 1].type == TT_EQ) or (self.current_token.type == TT_KEYWORD and self.current_token.value == 'static'):
            return res.success(res.register(self.identifier()))
        else:
            return self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or')))
    
    def if_expr(self):
        res = ParseResult()
        pos_start = self.current_token.pos_start
        if_token = self.current_token
        res.register(self.increment())
        
        if_expr_value = []
        if_expr_value.append(res.register(self.atom()))
        if res.error: return res
        
        if_expr_value.append(res.register(
            self.code_block((
                (TT_KEYWORD, 'elif'), 
                (TT_KEYWORD, 'else'), 
                (TT_KEYWORD, 'end')
            ), if_token.pos_start, if_token.pos_end)
        ))
        if res.error: return res
        
        elif_expr_values = []
        while self.current_token.type == TT_KEYWORD and self.current_token.value == 'elif':
            elif_expr_value = []
            elif_token = self.current_token
            res.register(self.increment())
            elif_expr_value.append(res.register(self.atom()))
            if res.error: return res
            
            elif_expr_value.append(res.register(
                self.code_block((
                    (TT_KEYWORD, 'elif'), 
                    (TT_KEYWORD, 'else'), 
                    (TT_KEYWORD, 'end')
                ), elif_token.pos_start, elif_token.pos_end)
            ))
            if res.error: return res
            elif_expr_values.append(elif_expr_value)
        
        else_expr_value = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'else':
            else_token = self.current_token
            res.register(self.increment())
            
            else_expr_value = (res.register(
                self.code_block((
                    (TT_KEYWORD, 'end'),
                ), else_token.pos_start, else_token.pos_end)
            ))
            if res.error: return res
        pos_end = self.current_token.pos_end
        res.register(self.increment())
        return res.success(IfNode(if_expr_value, elif_expr_values, else_expr_value, pos_start, pos_end))
    
    def type_choice(self):
        res = ParseResult()
        res.register(self.increment())
        
        if self.current_token.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected an identifier name"
            ))
        identifier = self.current_token
        res.register(self.increment())
        if self.current_token.type != TT_LT:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '<'"
            ))
        res.register(self.increment())
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'int':
            type_ = self.current_token
        elif self.current_token.type == TT_KEYWORD and self.current_token.value == 'float':
            type_ = self.current_token
        elif self.current_token.type == TT_KEYWORD and self.current_token.value == 'bool':
            type_ = self.current_token
        else:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected 'int' or 'float'"
            ))
        res.register(self.increment())
        if self.current_token.type != TT_GT:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '>'"
            ))
        res.register(self.increment())
        expr = res.register(self.statement())
        if res.error: return res
        
        return res.success(TypeChoiceNode(identifier, type_, expr))
    
    def statement(self):
        res = ParseResult()
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'if':
            statement = res.register(self.if_expr())
        elif self.current_token.type == TT_KEYWORD and self.current_token.value == 'choice':
            statement = res.register(self.type_choice())
        else:
            statement = res.register(self.expr())
        if res.error: return res
        return res.success(StatementNode(statement))
    
    def code(self):
        res = ParseResult()
        statements = res.register(self.code_block((TT_EOF,)))
        if res.error: return res

        return res.success(CodeNode(statements))
    
    #######################################################
    
    def bin_op(self, func_a, ops, func_b=None):
        if func_b is None: func_b = func_a
        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_token.type in ops or (self.current_token.type, self.current_token.value) in ops:
            op_token = self.current_token
            res.register(self.increment())
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_token, right)
            
        return res.success(left)
    
    def code_block(self, ends, pos_start=None, pos_end=None):
        statements = []
        res = ParseResult()
        if self.current_token.type == TT_NEWLINE:
            res.register(self.increment())
        
        statements.append(res.register(self.statement()))
        if res.error: return res
        
        while True:
            if self.current_token.type in ends or (self.current_token.type, self.current_token.value) in ends: break
            try:
                if self.tokens[self.token_index+1].type in ends or (self.tokens[self.token_index+1].type, self.tokens[self.token_index+1].value) in ends: break
            except IndexError:  
                if self.current_token.type == TT_EOF:
                    return res.failure(InvalidSyntaxError(
                        pos_start, pos_end,
                        "This code-block starting have no ending"
                    ))
            
            if self.current_token.type != TT_NEWLINE:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected '\\n' or ';'"
                ))
            res.register(self.increment())
            
            statements.append(res.register(self.statement()))
            if res.error: return res
        
        if self.current_token.type == TT_NEWLINE:
            res.register(self.increment())
        return res.success(statements)

### Nodes ###
class IntNode:
    def __init__(self, value: int):
        self.value = value
        self.type = 'int'
    
    def __repr__(self):
        return f'{self.value}'

class DoubleNode:
    def __init__(self, value: float):
        self.value = value
        self.type = 'double'
    
    def __repr__(self):
        return f'{self.value}'

class TrueNode:
    def __init__(self):
        self.value = 1
        self.type = '_Bool'
    
    def __repr__(self):
        return f'{self.value}'

class FalseNode:
    def __init__(self):
        self.value = 0
        self.type = '_Bool'
    
    def __repr__(self):
        return f'{self.value}'

class SymbolNode:
    def __init__(self, symbols, name):
        self.symbols = symbols
        self.name = name
        self.type = self.symbols[self.name].type
        self.symbols[self.name].usage_count += 1
        self.symbol_usage = self.symbols[self.name].usage_count
        self.type_choice = self.symbols[self.name].type_choice
        if isinstance(self.type, list) and self.type_choice is not None:
            self.type = self.type_choice
    
    def __repr__(self):
        self.symbol = self.symbols[self.name]
        if isinstance(self.symbol, Variable): return f'{self.symbol.name}_'
        if isinstance(self.type, list):
            result = f'identifiers[{self.symbol.identifier}]'
        else: result = f'*({self.type}*)identifiers[{self.symbol.identifier}]'
        if self.symbol_usage == self.symbol.usage_count:
            return result + f'\\*after:free(identifiers[{self.symbol.identifier}])*\\'
        return result

class SymbolAssignNode:
    def __init__(self, symbols, name, node, type_change=False):
        self.symbols = symbols
        self.name = name
        self.node = node
        self.type_change = type_change
        self.type = node.type
        self.symbol_type = self.symbols[self.name].type
        self.assign_count = self.symbols[self.name].assign_count
        self.usage_till_now = self.symbols[self.name].usage_count
    
    def __repr__(self):
        self.symbol = self.symbols[self.name]
        if self.symbol.usage_count - self.usage_till_now == 0:
            return f'{self.node}'
        
        if isinstance(self.symbol, Variable): return f'{self.symbol.name}_ = {self.node}'
        
        if self.type_change:
            if isinstance(self.symbol_type, list):
                return f'*({self.node.type}*)identifiers[{self.symbol.identifier}] = {self.node}\\*before:identifiers[{self.symbol.identifier}] = realloc(identifiers[{self.symbol.identifier}], sizeof({self.node.type}));after:{self.symbol.name}_type = {self.symbol.type.index(self.node.type)}*\\'
            return f'*({self.type}*)identifiers[{self.symbol.identifier}] = {self.node}\\*before:identifiers[{self.symbol.identifier}] = realloc(identifiers[{self.symbol.identifier}], sizeof({self.type}))*\\'
        
        if self.assign_count == 1:
            if isinstance(self.symbol.type, list):
                return f'*({self.type}*)identifiers[{self.symbol.identifier}] = {self.node}\\*before:identifiers[{self.symbol.identifier}] = malloc(sizeof({self.type}));before:int {self.symbol.name}_type = {self.symbol_type.index(self.type)}*\\'
            return f'*({self.type}*)identifiers[{self.symbol.identifier}] = {self.node}\\*before:identifiers[{self.symbol.identifier}] = malloc(sizeof({self.type}))*\\'
        return f'*({self.type}*)identifiers[{self.symbol.identifier}] = {self.node}'

class NegateNode:
    def __init__(self, node):
        self.node = node
        self.type = self.node.type
    
    def __repr__(self):
        return f'-({self.node})'

class AddNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        if self.left_node == 'double' or self.right_node == 'double': self.type = 'double'
        else: self.type = 'int'
    
    def __repr__(self):
        return f'({self.left_node})+({self.right_node})'

class SubtractNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        if self.left_node == 'double' or self.right_node == 'double': self.type = 'double'
        else: self.type = 'int'
    
    def __repr__(self):
        return f'({self.left_node})-({self.right_node})'

class MultiplyNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        if self.left_node == 'double' or self.right_node == 'double': self.type = 'double'
        else: self.type = 'int'
    
    def __repr__(self):
        return f'({self.left_node})*({self.right_node})'

class DivideNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = 'double'
    
    def __repr__(self):
        return f'({self.left_node})/(double)({self.right_node})'

class FunctionCallNode:
    def __init__(self, name, type_, arguments):
        self.name = name
        self.type = type_
        self.arguments = arguments
    
    def __repr__(self):
        arg_str = ''
        first = True
        for argument in self.arguments:
            if first: arg_str += f'{argument}'
            else: arg_str += f', {argument}'
            first = False
        
        return f'({self.type}){self.name}({arg_str})'

class EqualNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})==({self.right_node})'

class NotEqualNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
    
    def __repr__(self):
        return f'({self.left_node})!=({self.right_node})'

class LessThanNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})<({self.right_node})'

class GreaterThanNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})>({self.right_node})'

class LessThanEqualNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})<=({self.right_node})'

class GreaterThanEqualNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})>=({self.right_node})'

class AndNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})&&({self.right_node})'

class OrNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'({self.left_node})||({self.right_node})'

class NotNode:
    def __init__(self, node):
        self.node = node
        self.type = '_Bool'
    
    def __repr__(self):
        return f'!({self.node})'

class CIfNode:
    def __init__(self, if_value, elif_values=[], else_value=None):
        self.if_value = if_value
        self.elif_values = elif_values
        self.else_value = else_value
    
    def commandize(self, cmds, statement_str):
        result = statement_str
        for cmd in cmds:
            cmd = cmd.replace('\\*', '')
            cmd = cmd.replace('*\\', '')
            for command in cmd.split(';'):
                if not command: continue
                head, line = command.split(':')
                if head == 'before':
                    result = f'{line};\n{result}'
                if head == 'after':
                    result = f'{result};\n{line}'
                if head == 'pre-before':
                    result += f'\\*before:{line}*\\'
        result += ';\n'
        return result
    
    def code_block(self, statements):
        result = ''
        for statement in statements:
            statement_str = statement.__repr__()
            if len(statement_str) < 2:
                result += statement_str + ';\n'
                continue
            
            cmd = []
            while '\\*' in statement_str:
                tmpi = statement_str.find('\\*')
                tmp = statement_str[tmpi:statement_str.find('*\\', tmpi)+2]
                statement_str = statement_str.replace(tmp, '')
                cmd.append(tmp)
            result += self.commandize(cmd, statement_str)
        return result
    
    def __repr__(self):
        result = f'if ({self.if_value[0]}){{\n'
        result += self.code_block(self.if_value[1])
        result += '}'
        
        for elif_value in self.elif_values:
            result += f'else if ({elif_value[0]}){{\n'
            result += self.code_block(elif_value[1])
            result += '}'
        
        if self.else_value:
            result += 'else{\n'
            result += self.code_block(self.else_value)
            result += '}'
        
        return result

class CStatementNode:
    def __init__(self, node):
        self.node = node
    
    def __repr__(self):
        return f'{self.node}'

class GlobalVariableNode:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_
    
    def __repr__(self):
        return f'{self.type} {self.name};\n'

class CCodeNode:
    def __init__(self, libraries, identifier_count, global_variables, statement_nodes):
        self.libraries = libraries
        self.identifier_count = identifier_count
        self.global_variables = global_variables
        self.statement_nodes = statement_nodes
    
    def commandize(self, cmds, statement_str):
        result = statement_str
        for cmd in cmds:
            cmd = cmd.replace('\\*', '')
            cmd = cmd.replace('*\\', '')
            for command in cmd.split(';'):
                if not command: continue
                head, line = command.split(':')
                if head == 'before':
                    result = f'{line};\n{result}'
                if head == 'after':
                    result = f'{result};\n{line}'
        result += ';\n'
        return result
    
    def __repr__(self):
        result = ''
        for library in self.libraries:
            result += f'{library}'
        if self.identifier_count:
            result += f'\nvoid* identifiers[{self.identifier_count}];\n'
        for variable in self.global_variables:
            if variable.usage_count:
                result += f'{variable.type} {variable.name}_;\n'
        result += 'int main(){\n'
        for statement in self.statement_nodes:
            statement_str = statement.__repr__()
            if len(statement_str) < 2:
                result += statement_str + ';\n'
                continue
            
            cmd = []
            while '\\*' in statement_str:
                tmpi = statement_str.find('\\*')
                tmp = statement_str[tmpi:statement_str.find('*\\', tmpi)+2]
                statement_str = statement_str.replace(tmp, '')
                cmd.append(tmp)
            result += self.commandize(cmd, statement_str)
        result += 'return 0;\n}'
        return result

### AnalizeResult ###
class AnalizeResult:
    def __init__(self):
        self.value = None
        self.error = None
    
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self

class Symbol:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_
        self.assign_count = 0
        self.usage_count = 0
        self.manual_static = False
        self.type_choice = None

class Identifier(Symbol):
    def __init__(self, name, type_, identifier, start_assign_count, start_usage_count, type_choice):
        super().__init__(name, type_)
        self.identifier = identifier
        self.first_type = self.type
        self.assign_count = start_assign_count
        self.usage_count = start_usage_count
        self.if_thingy = 0
        self.type_choice = type_choice

class Variable(Symbol):
    def __init__(self, name, type_, manual_static):
        super().__init__(name, type_)
        self.manual_static = manual_static
    
    def convert_to_identifier(self, identifier):
        if self.manual_static:
            return
        return Identifier(self.name, self.type, identifier, self.assign_count, self.usage_count, self.type_choice)

### SymbolTable ###
class SymbolTable:
    BRANCH_IF = 0
    BRANCH_ELSE = 1
    def __init__(self):
        self.branchs = []
        self.branch_count = 0
        self.symbols = {}
        self.identifier_count = 0
        self.global_variables = []
    
    def symbol_get(self, name, node):
        res = AnalizeResult()
        if name in self.symbols.keys():
            return res.success(SymbolNode(self.symbols, name))
        return res.failure(
            NameError_(node.pos_start, node.pos_end, f"Name '{name}' is not defined")
        )
    
    def symbol_assign(self, name, node, value_node, manual_static, libraries):
        res = AnalizeResult()
        if name in self.symbols.keys():
            symbol = self.symbols[name]
            
            if isinstance(symbol, Identifier) and manual_static:
                return res.failure(
                    TypeError_(node.manual_static_token.pos_start, node.manual_static_token.pos_end,
                               "Can't make a dynamic variable static"
                ))
            
            if value_node.type == symbol.type:
                if manual_static: symbol.manual_static = True
                symbol.assign_count += 1
                return res.success(SymbolAssignNode(self.symbols, name, value_node))
            
            if symbol.manual_static or manual_static:
                return res.failure(
                    TypeError_(node.pos_start, node.pos_end, 
                    f"Can't change static variables's type"
                ))
            
            if isinstance(symbol, Variable):
                self.global_variables.remove(symbol)
                symbol = symbol.convert_to_identifier(self.identifier_count)
                self.symbols[name] = symbol
                self.identifier_count += 1
                libraries.add('#include<stdlib.h>\n')
            if self.branchs:
                if self.branchs[-1] == SymbolTable.BRANCH_IF:
                    if isinstance(symbol.type, list):
                        if value_node.type not in symbol.type:
                            symbol.type.append(value_node.type)
                    else: symbol.type = [symbol.type, value_node.type]
                    symbol.if_thingy == self.branch_count
                else:
                    if isinstance(symbol.type, list):
                        if value_node.type not in symbol.type:
                            if symbol.if_thingy +1 == self.branch_count:
                                symbol.type[0] = value_node.type
                            else: symbol.type.append(value_node.type)
                    else: symbol.type = [symbol.type, value_node.type]
            else:
                symbol.type = value_node.type
            symbol.assign_count += 1
            return res.success(SymbolAssignNode(self.symbols, name, value_node, type_change=True))
        
        if self.branchs:
            return res.failure(NameError_(
                node.pos_start, node.pos_end,
                "Can't initiate an identifier inside a branch"
            ))
        symbol = Variable(name, value_node.type, manual_static)
        self.symbols[name] = symbol
        self.global_variables.append(symbol)
        symbol.assign_count += 1
        return res.success(SymbolAssignNode(self.symbols, name, value_node))

    def symbol_type_choice(self, name, type_, node=None):
        type_map = {'int':'int', 'bool':'_Bool', 'float':'double'}
        
        res = AnalizeResult()
        if node is not None:
            if name not in self.symbols.keys():
                return res.failure(
                    NameError_(node.pos_start, node.pos_end, f"Name '{name}' is not defined")
                )
            symbol = self.symbols[name]
            if not isinstance(symbol.type, list):
                return res.failure(
                    TypeError_(node.pos_start, node.pos_end, "Can't type choice on single branch variables")
                )
            if type_map[type_.value] not in symbol.type:
                return res.failure(
                    TypeError_(type_.pos_start, type_.pos_end, f"Identifier '{name}' does not have '{type_.value}' type branch")
                )
            self.symbols[name].type_choice = type_map[type_.value]
            return res.success(SymbolNode(self.symbols, name))
        self.symbols[name].type_choice = type_
        return res.success(SymbolNode(self.symbols, name))
    
    def symbol_type_choice_end(self, name):
        self.symbols[name].type_choice = None
    
    def start_if_branch(self):
        self.branch_count += 1
        self.branchs.append(SymbolTable.BRANCH_IF)
    
    def start_else_branch(self):
        self.branchs.append(SymbolTable.BRANCH_ELSE)
    
    def end_branch(self):
        self.branchs.pop()

### Analizer ###
class Analizer:
    def __init__(self):
        self.libraries = set()
    
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_node)
        return method(node)
    
    def no_visit_node(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined.')
    
    def visit_NumberNode(self, node):
        res = AnalizeResult()
        if type(node.token.value) is int: return res.success(IntNode(node.token.value))
        if type(node.token.value) is float: return res.success(DoubleNode(node.token.value))
    
    def visit_KeywordValueNode(self, node):
        res = AnalizeResult()
        if node.token.value == 'True':
            return res.success(TrueNode())
        if node.token.value == 'False':
            return res.success(FalseNode())
    
    def visit_IdentifierNode(self, node):
        return self.symbol_table.symbol_get(node.id_name_token.value, node)
    
    def visit_IdAssignNode(self, node):
        res = AnalizeResult()
        value_node = res.register(self.visit(node.value_node))
        if res.error: return res
        if isinstance(value_node.type, list):
            if_expr = []
            elif_expr = []
            symbol = res.register(self.symbol_table.symbol_get(value_node.name, node))
            for i, value_type in enumerate(value_node.type):
                if not i:
                    self.symbol_table.start_if_branch()
                    if_expr.append(
                        EqualNode(
                            f'{symbol.name}_type', 
                            IntNode(symbol.type.index(value_type))))
                    res.register(self.symbol_table.symbol_type_choice(value_node.name, value_type))
                    if_expr.append([res.register(self.symbol_table.symbol_assign(
                            node.id_name_token.value,
                            node, res.register(self.visit(node.value_node)),
                            node.manual_static,
                            self.libraries))])
                    self.symbol_table.end_branch()
                else:
                    tmpelif_expr = []
                    self.symbol_table.start_else_branch()
                    self.symbol_table.start_if_branch()
                    tmpelif_expr.append(
                        EqualNode(
                            f'{symbol.name}_type', 
                            IntNode(symbol.type.index(value_type))))
                    res.register(self.symbol_table.symbol_type_choice(value_node.name, value_type))
                    tmpelif_expr.append([res.register(self.symbol_table.symbol_assign(
                            node.id_name_token.value,
                            node, res.register(self.visit(node.value_node)),
                            node.manual_static,
                            self.libraries))])
                    elif_expr.append(tmpelif_expr)
                    self.symbol_table.end_branch()
                    self.symbol_table.end_branch()
                if res.error: return res
                self.symbol_table.symbol_type_choice_end(value_node.name)
            return res.success(CIfNode(if_expr, elif_expr))
        
        name = node.id_name_token.value
        answer = self.symbol_table.symbol_assign(name, node, value_node, node.manual_static, self.libraries)
        return answer
    
    def visit_IfNode(self, node):
        res = AnalizeResult()
        if_value_node = []
        self.symbol_table.start_if_branch()
        if_value_node.append(res.register(self.visit(node.if_value[0])))
        if res.error: return res
        
        if_statement_nodes = []
        for statement in node.if_value[1]:
            if_statement_nodes.append(res.register(self.visit(statement)))
            if res.error: return res
        if_value_node.append(if_statement_nodes)
        self.symbol_table.end_branch()
        elif_value_nodes = []
        for elif_value in node.elif_values:
            self.symbol_table.start_else_branch()
            self.symbol_table.start_if_branch()
            elif_expr, elif_statements = elif_value
            elif_value_node = []
            elif_value_node.append(res.register(self.visit(elif_expr)))
            if res.error: return res
            elif_statement_nodes = []
            for statement in elif_statements:
                elif_statement_nodes.append(res.register(self.visit(statement)))
                if res.error: return res
            elif_value_node.append(elif_statement_nodes)
            elif_value_nodes.append(elif_value_node)
            self.symbol_table.end_branch()
            self.symbol_table.end_branch()
        
        else_value_node = []
        if node.else_value is not None:
            self.symbol_table.start_else_branch()
            for statement in node.else_value:
                else_value_node.append(res.register(self.visit(statement)))
                if res.error: return res
            self.symbol_table.end_branch()
        return res.success(CIfNode(if_value_node, elif_value_nodes, else_value_node))
    
    def visit_TypeChoiceNode(self, node):
        map_type = {'int':'int', 'bool':'_Bool', 'float':'double'}
        res = AnalizeResult()
        symbol = res.register(self.symbol_table.symbol_get(node.identifier.value, node))
        if res.error: return res
        if_expr = []
        if_expr.append(EqualNode(
                f'{symbol.name}_type', 
                IntNode(symbol.type.index(map_type[node.type.value]))
            ))
        res.register(self.symbol_table.symbol_type_choice(node.identifier.value, node.type, node))
        expr = res.register(self.visit(node.expr))
        if res.error: return res
        if_expr.append([expr])
        self.symbol_table.symbol_type_choice_end(node.identifier.value)
        return res.success(CIfNode(if_expr))
    
    def visit_BinOpNode(self, node):
        res = AnalizeResult()
        left_node = res.register(self.visit(node.left_node))
        if res.error: return res
        right_node = res.register(self.visit(node.right_node))
        if res.error: return res
        
        if node.op_token.type == TT_PLUS:
            return res.success(AddNode(left_node, right_node))
        if node.op_token.type == TT_MINUS:
            return res.success(SubtractNode(left_node, right_node))
        if node.op_token.type == TT_MUL:
            return res.success(MultiplyNode(left_node, right_node))
        if node.op_token.type == TT_DIV:
            return res.success(DivideNode(left_node, right_node))
        if node.op_token.type == TT_POW:
            type_ = 'double' if left_node.type == 'double' or right_node.type == 'double' else 'int'
            self.libraries.add('#include<math.h>\n')
            return res.success(FunctionCallNode('pow', type_, (left_node, right_node)))
        if node.op_token.type == TT_EE:
            return res.success(EqualNode(left_node, right_node))
        if node.op_token.type == TT_EE:
            return res.success(NotEqualNode(left_node, right_node))
        if node.op_token.type == TT_LT:
            return res.success(LessThanNode(left_node, right_node))
        if node.op_token.type == TT_GT:
            return res.success(GreaterThanNode(left_node, right_node))
        if node.op_token.type == TT_LTE:
            return res.success(LessThanEqualNode(left_node, right_node))
        if node.op_token.type == TT_GTE:
            return res.success(GreaterThanEqualNode(left_node, right_node))
        if node.op_token.type == TT_KEYWORD and node.op_token.value == 'and':
            return res.success(AndNode(left_node, right_node))
        if node.op_token.type == TT_KEYWORD and node.op_token.value == 'or':
            return res.success(OrNode(left_node, right_node))
    
    def visit_UnaryOpNode(self, node):
        res = AnalizeResult()
        value_node = res.register(self.visit(node.node))
        if res.error: return res
        
        if node.op_token.type == TT_PLUS:
            return res.success(value_node)
        if node.op_token.type == TT_MINUS:
            return res.success(NegateNode(value_node))
        if node.op_token.type == TT_KEYWORD and node.op_token.value == 'or':
            return res.success(NotNode(value_node))
    
    def visit_StatementNode(self, node):
        res = AnalizeResult()
        statement_node = res.register(self.visit(node.statement_node))
        if res.error: return res
        return res.success(CStatementNode(statement_node))
    
    def visit_CodeNode(self, node):
        self.symbol_table = SymbolTable()
        res = AnalizeResult()
        statement_nodes = []
        for statement in node.statements:
            statement_nodes.append(res.register(self.visit(statement)))
            if res.error: return res
        return res.success(CCodeNode(self.libraries, self.symbol_table.identifier_count, self.symbol_table.global_variables, statement_nodes))

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
    
    c = analize(ast)
    if c.error: return None, c.error
    return c.value, None

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
                f.write(result.__repr__())
            try:
                subprocess.Popen(f"gcc -O2 {xfile+'.c'} -o {xfile+'.exe'}")
            except FileNotFoundError:
                print("gcc is not installed, you need gcc compiler to use this program ;(")
    
    elif cmd == 'run':
        result, error = run(file_name, context)
        if error: print(error.as_string())
        else:
            xfile = file_name.split('.')[0]
            with open(xfile+'.c', 'w') as f:
                f.write(result.__repr__())
            try:
                subprocess.Popen(f"gcc -O2 {xfile+'.c'} -o {xfile+'.exe'}")
                subprocess.Popen(f"./{xfile+'.exe'}")
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
