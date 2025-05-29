class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # '+', '*', 'a', etc.
        self.left = left    # 좌측 자식 (연산자나 심볼)
        self.right = right  # 우측 자식 (연산자나 심볼)

    def __repr__(self, level=0):
        ret = "  " * level + repr(self.value) + "\n"
        if self.left:
            ret += self.left.__repr__(level + 1)
        if self.right:
            ret += self.right.__repr__(level + 1)
        return ret

# + < . < *
# 우선순위 정의
precedence = {
    '*': 3,
    '.': 2,  # 접속 연산
    '+': 1
}

# 알파벳 또는 숫자 확인
def is_symbol(c):
    return c.isalnum()

# 입력 → 토큰 리스트로 변환
def tokenize(re):
    tokens = []
    prev = ''
    for i, c in enumerate(re):
        if prev and (
            (is_symbol(prev) or prev == ')' or prev == '*') and
            (is_symbol(c) or c == '(')
        ):
            tokens.append('.') #접속 명사 추가
        tokens.append(c)
        prev = c
    return tokens

# 파서 + 트리 생성
def parse(tokens):
    output = []
    ops = []

    def pop_op():
        op = ops.pop()
        if op == '*':
            right = output.pop()
            output.append(Node(op, right))  # 단항 연산
        else:
            right = output.pop()
            left = output.pop()
            output.append(Node(op, left, right))  # 이항 연산

    for token in tokens:
        if is_symbol(token):
            output.append(Node(token))
        elif token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                pop_op()
            ops.pop()  # '(' 지우기
        elif token in precedence:
            while (
                ops and ops[-1] in precedence and
                precedence[ops[-1]] >= precedence[token]
            ):
                pop_op()
            ops.append(token)

    while ops:
        pop_op()

    return output[0]  # 루트 리턴

#정규표현 입력 => 트리 생성
def build_tree_from_re(re):
    tokens = tokenize(re)
    return parse(tokens)

#트리 구조를 딕셔너리 형태로 변환
def tree_to_dict(node):
    if node is None:
        return None
    result = {"name": node.value}
    children = []
    if node.left:
        children.append(tree_to_dict(node.left))
    if node.right:
        children.append(tree_to_dict(node.right))
    if children:
        result["children"] = children
    return result

def traverse_and_count(node):
    result = {
        'operator_total': 0,
        'operator_plus': 0,
        'operator_star': 0,
        'operator_dot': 0,
        'operand': 0
    }

    def dfs(n):
        if n is None:
            return
        if n.value in {'+', '*', '.'}:
            result['operator_total'] += 1
            if n.value == '+':
                result['operator_plus'] += 1
            elif n.value == '*':
                result['operator_star'] += 1
            elif n.value == '.':
                result['operator_dot'] += 1
        else:
            result['operand'] += 1

        dfs(n.left)
        dfs(n.right)

    dfs(node)
    return result

# 전체 size
def size(counts):
    return counts['operator_total'] + counts['operand']

# arc 상한 개수
def size_max_arc(counts):
    return size(counts) * 4

# 정확한 arc 수 계산
def size_acc_arc(counts):
    return (counts['operator_plus'] * 4 +
            counts['operator_dot'] * 2 +
            counts['operator_star'] * 3)


import json

# aA+b+0c*
if __name__ == "__main__":
    re = "aA+b+0c*"
    tree = build_tree_from_re(re)
    tree_json = tree_to_dict(tree)
    counts = traverse_and_count(tree)

    print("정규 표현 : ", re)
    print("size :", size(counts))
    print("arc 상한 :", size_max_arc(counts))
    print("실제 arc 수 :", size_acc_arc(counts))
    
    # D3에서 사용할 JSON 저장
    with open(f"tree.json", "w") as f:
        json.dump(tree_json, f, indent=2)
        