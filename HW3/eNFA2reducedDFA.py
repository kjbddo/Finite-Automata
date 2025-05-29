from typing import Set, Dict, List
import numpy as np
from collections import defaultdict
import RE2Tree
from Tree2eNFA import tree_to_eNFA

class ENFA:
    def __init__(self, states: Set[str], alphabet: Set[str], start_state: str, 
                 final_states: Set[str], transitions: Dict[str, Dict[str, Set[str]]]):
        self.states = states
        self.alphabet = alphabet
        self.start_state = start_state
        self.final_states = final_states
        self.transitions = transitions

class DFA:
    def __init__(self, states: Set[str], alphabet: Set[str], start_state: str,
                 final_states: Set[str], transitions: Dict[str, Dict[str, str]]):
        self.states = states
        self.alphabet = alphabet
        self.start_state = start_state
        self.final_states = final_states
        self.transitions = transitions

def convert_tree2enfa_output_to_enfa(enfa_dict):
    """Tree2eNFA의 출력을 ENFA 객체로 변환합니다."""
    states = set(map(str, enfa_dict['states']))
    alphabet = set(enfa_dict['alphabet'])
    start_state = str(enfa_dict['start_state'])
    final_states = set(map(str, enfa_dict['final_states']))
    
    # 전이 함수 변환
    transitions = defaultdict(lambda: defaultdict(set))
    for i, from_state in enumerate(enfa_dict['states']):
        for j, to_state in enumerate(enfa_dict['states']):
            if enfa_dict['transitions'][i][j]:
                for symbol in enfa_dict['transitions'][i][j]:
                    transitions[str(from_state)][symbol].add(str(to_state))
    
    return ENFA(states, alphabet, start_state, final_states, transitions)

def epsilon_closure(enfa: ENFA, states: Set[str]) -> Set[str]:
    """주어진 상태들의 ε-클로저를 계산합니다."""
    closure = states.copy()
    stack = list(states)
    
    while stack:
        state = stack.pop()
        if state in enfa.transitions and 'ε' in enfa.transitions[state]:
            for next_state in enfa.transitions[state]['ε']:
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
    
    return closure

def move(enfa: ENFA, states: Set[str], symbol: str) -> Set[str]:
    """주어진 상태들에서 특정 심볼로 이동 가능한 모든 상태를 반환합니다."""
    result = set()
    for state in states:
        if state in enfa.transitions and symbol in enfa.transitions[state]:
            result.update(enfa.transitions[state][symbol])
    return result

def enfa_to_dfa(enfa: ENFA) -> DFA:
    """ε-NFA를 DFA로 변환합니다."""
    # 초기 상태의 ε-클로저 계산
    initial_states = epsilon_closure(enfa, {enfa.start_state})
    
    # DFA의 상태와 전이 함수 초기화
    dfa_states = set()
    dfa_transitions = defaultdict(dict)
    unprocessed_states = [initial_states]
    dfa_states.add(frozenset(initial_states))
    
    # 시작 상태와 종결 상태 설정
    dfa_start_state = frozenset(initial_states)
    dfa_final_states = set()
    
    # 종결 상태 확인
    if any(state in enfa.final_states for state in initial_states):
        dfa_final_states.add(dfa_start_state)
    
    while unprocessed_states:
        current_states = unprocessed_states.pop(0)
        
        for symbol in enfa.alphabet:
            # 현재 상태에서 심볼로 이동 가능한 상태들의 ε-클로저 계산
            next_states = epsilon_closure(enfa, move(enfa, current_states, symbol))
            
            if next_states:
                if frozenset(next_states) not in dfa_states:
                    dfa_states.add(frozenset(next_states))
                    unprocessed_states.append(next_states)
                    
                    # 종결 상태 확인
                    if any(state in enfa.final_states for state in next_states):
                        dfa_final_states.add(frozenset(next_states))
                
                dfa_transitions[frozenset(current_states)][symbol] = frozenset(next_states)
    
    return DFA(dfa_states, enfa.alphabet, dfa_start_state, dfa_final_states, dfa_transitions)

def minimize_dfa(dfa: DFA) -> DFA:
    """DFA를 최소화합니다."""
    # 도달 가능한 상태 찾기
    reachable_states = {dfa.start_state}
    stack = [dfa.start_state]
    
    while stack:
        state = stack.pop()
        for symbol in dfa.alphabet:
            if symbol in dfa.transitions[state]:
                next_state = dfa.transitions[state][symbol]
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    stack.append(next_state)
    
    # 도달 가능한 상태만 사용
    dfa.states = reachable_states
    dfa.final_states = dfa.final_states.intersection(reachable_states)
    
    # 상태 분할
    partitions = [dfa.final_states, dfa.states - dfa.final_states]
    partitions = [p for p in partitions if p]  # 빈 집합 제거
    
    while True:
        new_partitions = []
        for partition in partitions:
            if len(partition) <= 1:
                new_partitions.append(partition)
                continue
            
            # 분할 기준 찾기
            split_groups = defaultdict(set)
            for state in partition:
                key = tuple(dfa.transitions[state].get(symbol, None) for symbol in dfa.alphabet)
                split_groups[key].add(state)
            
            new_partitions.extend(split_groups.values())
        
        if len(new_partitions) == len(partitions):
            break
        partitions = new_partitions
    
    # 새로운 DFA 생성
    new_states = set()
    new_transitions = defaultdict(dict)
    new_final_states = set()
    state_mapping = {}
    
    for partition in partitions:
        new_state = frozenset(partition)
        new_states.add(new_state)
        state_mapping[new_state] = partition
        
        if partition.intersection(dfa.final_states):
            new_final_states.add(new_state)
    
    # 시작 상태 찾기
    new_start_state = None
    for new_state, old_states in state_mapping.items():
        if dfa.start_state in old_states:
            new_start_state = new_state
            break
    
    # 전이 함수 생성
    for new_state, old_states in state_mapping.items():
        for symbol in dfa.alphabet:
            if any(symbol in dfa.transitions[state] for state in old_states):
                next_states = set()
                for state in old_states:
                    if symbol in dfa.transitions[state]:
                        next_states.add(dfa.transitions[state][symbol])
                
                for partition in partitions:
                    if next_states.issubset(partition):
                        new_transitions[new_state][symbol] = frozenset(partition)
                        break
    
    return DFA(new_states, dfa.alphabet, new_start_state, new_final_states, new_transitions)

def dfa_to_dict(dfa: DFA):
    """DFA를 Tree2eNFA 형식의 딕셔너리로 변환합니다."""
    # 상태를 정수로 매핑
    state_to_int = {state: i for i, state in enumerate(sorted(dfa.states))}
    
    # 전이 행렬 초기화
    n_states = len(dfa.states)
    transition_matrix = [[[] for _ in range(n_states)] for _ in range(n_states)]
    
    # 전이 함수 채우기
    for from_state in dfa.states:
        for symbol in dfa.alphabet:
            if symbol in dfa.transitions[from_state]:
                to_state = dfa.transitions[from_state][symbol]
                from_idx = state_to_int[from_state]
                to_idx = state_to_int[to_state]
                if symbol not in transition_matrix[from_idx][to_idx]:
                    transition_matrix[from_idx][to_idx].append(symbol)
    
    return {
        'states': list(range(n_states)),
        'alphabet': sorted(list(dfa.alphabet)),
        'start_state': state_to_int[dfa.start_state],
        'transitions': transition_matrix,
        'final_states': [state_to_int[state] for state in sorted(dfa.final_states)]
    }

def print_dfa(dfa: DFA):
    """DFA의 구성 요소를 Tree2eNFA 형식으로 출력합니다."""
    dfa_dict = dfa_to_dict(dfa)
    
    print("\n=== DFA 구성 요소 ===")
    print(f"상태: {dfa_dict['states']}")
    print(f"입력 심볼 집합: {dfa_dict['alphabet']}")
    print(f"시작 상태: {dfa_dict['start_state']}")
    print(f"종결 상태: {dfa_dict['final_states']}")
    
    print("\n전이 함수:")
    for i in range(len(dfa_dict['transitions'])):
        for j in range(len(dfa_dict['transitions'][i])):
            if dfa_dict['transitions'][i][j]:
                print(f"{i} -> {dfa_dict['transitions'][i][j]} -> {j}")

def compare_states(enfa: ENFA, dfa: DFA, re: str):
    """ε-NFA와 DFA의 상태 수를 비교합니다."""
    print("\n=== 상태 수 비교 ===")
    print(f"정규표현식: {re}")
    print(f"ε-NFA 상태 수: {len(enfa.states)}")
    print(f"DFA 상태 수: {len(dfa.states)}")
    print(f"상태 수 감소율: {((len(enfa.states) - len(dfa.states)) / len(enfa.states) * 100):.2f}%")

def convert_to_reduced_dfa(enfa_dict):
    """Tree2eNFA의 출력을 받아 Reduced DFA로 변환합니다."""
    # ENFA 객체로 변환
    enfa = convert_tree2enfa_output_to_enfa(enfa_dict)
    
    # DFA로 변환
    dfa = enfa_to_dfa(enfa)
    
    # DFA 최소화
    reduced_dfa = minimize_dfa(dfa)
    
    return reduced_dfa

def visualize_dfa(dfa: DFA):
    """DFA를 테이블과 그래프로 시각화합니다."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd
    
    dfa_dict = dfa_to_dict(dfa)
    
    # 그래프와 표를 나란히 표시하기 위한 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가
    for state in dfa_dict['states']:
        G.add_node(state)
    
    # 엣지 추가
    for i in range(len(dfa_dict['transitions'])):
        for j in range(len(dfa_dict['transitions'][i])):
            if dfa_dict['transitions'][i][j]:
                G.add_edge(i, j)
    
    # 계층적 레이아웃 생성
    pos = {}
    
    # 시작 상태를 가장 왼쪽에 배치
    pos[dfa_dict['start_state']] = (0, 0)
    
    # 시작 상태로부터의 거리에 따라 x 좌표 조정
    distances = nx.single_source_shortest_path_length(G, dfa_dict['start_state'])
    max_distance = max(distances.values())
    
    # 각 거리별로 노드들을 수직으로 정렬
    nodes_by_distance = {}
    for node, distance in distances.items():
        if distance not in nodes_by_distance:
            nodes_by_distance[distance] = []
        nodes_by_distance[distance].append(node)
    
    # 종결 상태를 가장 오른쪽에 배치
    for final_state in dfa_dict['final_states']:
        pos[final_state] = (max_distance + 1, 0)
    
    # 각 거리별로 노드들을 수직으로 배치
    for distance in range(1, max_distance + 1):
        nodes = nodes_by_distance.get(distance, [])
        # 종결 상태는 제외
        nodes = [node for node in nodes if node not in dfa_dict['final_states']]
        nodes.sort()  # 노드 번호로 정렬
        
        # 수직 간격 계산 
        vertical_spacing = 2.0
        if len(nodes) > 1:
            total_height = (len(nodes) - 1) * vertical_spacing
            start_y = -total_height / 2
            
            for i, node in enumerate(nodes):
                y = start_y + i * vertical_spacing
                pos[node] = (distance, y)
        elif len(nodes) == 1:
            # 단일 노드인 경우 중앙에 배치
            pos[nodes[0]] = (distance, 0)
    
    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.7,
                          ax=ax1)
    
    # 시작 상태와 종결 상태 강조
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[dfa_dict['start_state']],
                          node_color='green',
                          node_size=1000,
                          alpha=0.7,
                          ax=ax1)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=dfa_dict['final_states'],
                          node_color='red',
                          node_size=1000,
                          alpha=0.7,
                          ax=ax1)
    
    # 엣지 그리기 
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          connectionstyle='arc3,rad=0.2',
                          ax=ax1)
    
    # 레이블 그리기
    labels = {}
    for state in G.nodes():
        # 해당 상태에서 나가는 전이들의 심볼들을 수집
        symbols = set()
        for j in range(len(dfa_dict['transitions'][state])):
            if dfa_dict['transitions'][state][j]:
                symbols.update(dfa_dict['transitions'][state][j])
        # 상태 레이블 생성 (심볼이 있는 경우에만 표시)
        label = f"q{state}"
        if symbols:
            label += f"\n({','.join(sorted(symbols))})"
        labels[state] = label

    nx.draw_networkx_labels(G, pos, 
                           labels=labels,
                           font_size=10,
                           ax=ax1)
    
    # 엣지 레이블 그리기 
    edge_labels = {}
    for i in range(len(dfa_dict['transitions'])):
        for j in range(len(dfa_dict['transitions'][i])):
            if dfa_dict['transitions'][i][j]:
                edge_labels[(i, j)] = ','.join(dfa_dict['transitions'][i][j])
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax1)
    
    ax1.set_title("DFA 그래프")
    ax1.axis('off')
    
    # 전이 함수 표 생성
    states = sorted(dfa_dict['states'])
    alphabet = sorted(dfa_dict['alphabet'])
    
    # 전이 함수 표 생성
    transition_table = []
    for state in states:
        row = []
        for symbol in alphabet:
            # 현재 상태에서 해당 심볼로 전이 가능한 모든 상태 찾기
            next_states = []
            for next_state in states:
                if dfa_dict['transitions'][state][next_state] and symbol in dfa_dict['transitions'][state][next_state]:
                    next_states.append(f"q{next_state}")
            row.append(','.join(next_states) if next_states else '-')
        transition_table.append(row)
    
    # pandas DataFrame 생성
    df = pd.DataFrame(transition_table, 
                     index=[f"q{state}" for state in states],
                     columns=alphabet)
    
    # 표 시각화
    ax2.axis('off')
    table = ax2.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.1] * len(df.columns))
    
    # 표 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 모든 셀의 스타일 설정
    for cell in table._cells:
        table._cells[cell].set_text_props(wrap=True)
    
    # 시작 상태와 종결 상태 강조
    for i, state in enumerate(states):
        if state == dfa_dict['start_state']:
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('#90EE90')  # 밝은 초록색
        if state in dfa_dict['final_states']:
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('#FFB6C1')  # 밝은 빨간색
    
    # 헤더 스타일 설정
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#E6E6E6')  # 회색 배경
        table[(0, j)].set_text_props(weight='bold')
    
    ax2.set_title("DFA 전이 함수 표", fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()

def main():
    re = "aA+b+0c*"
    tree = RE2Tree.build_tree_from_re(re)
    eNFA = tree_to_eNFA(tree)
    #Reduced DFA 변환
    reduced_dfa = convert_to_reduced_dfa(eNFA)
    print_dfa(reduced_dfa)
    
    enfa = convert_tree2enfa_output_to_enfa(eNFA)
    #상태 수 비교
    compare_states(enfa, reduced_dfa, re)
    
    # DFA 시각화
    visualize_dfa(reduced_dfa)
    
    #여러 테스트 케이스들에 대한 상태 수 비교
    test_regexes = [
        "a(b+c)*", "a(bc+de)", "a*+b*", "(a+b)c", "a(b+c)d", "a(bc)*d", "(a+b+c)*", "a(b+c+d)*e", "((a+b)*c*)*d"
    ]
    
    for re in test_regexes:
        tree = RE2Tree.build_tree_from_re(re)
        eNFA = tree_to_eNFA(tree)
        enfa = convert_tree2enfa_output_to_enfa(eNFA)
        compare_states(enfa, reduced_dfa, re)

if __name__ == "__main__":
    main()
