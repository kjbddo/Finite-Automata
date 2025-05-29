#convert Tree to e-NFA
import RE2Tree 
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

#내부 구조로 상태 집합, 입력 심볼 집합, 시작 상태, 전이 함수, 종결 상태집합을 구성
#전이함수는 이차원 배열로 구현
#노드 탐색은 dfs로 구현
def tree_to_eNFA(tree):
    # 상태 번호를 관리하기 위한 전역 변수 
    state_counter = 0
    
    # e-NFA의 구성요소
    states = set()  # 상태 집합
    alphabet = set()  # 입력 심볼 집합
    start_state = None  # 시작 상태
    transitions = []  # 전이 함수 (2차원 배열)
    final_states = set()  # 종결 상태 집합
    
    def create_state():
        nonlocal state_counter
        state = state_counter
        state_counter += 1
        states.add(state)
        return state
    
    def dfs(node):
        if node is None:
            return None, None
            
        if node.value.isalnum():  # 심볼인 경우
            start = create_state()
            end = create_state()
            alphabet.add(node.value)
            transitions.append([start, node.value, end])
            return start, end
            
        elif node.value == '+':  # 합집합
            start = create_state()
            end = create_state()
            
            # 왼쪽 서브트리 처리
            left_start, left_end = dfs(node.left)
            if left_start is not None:
                transitions.append([start, 'ε', left_start])
                transitions.append([left_end, 'ε', end])
            
            # 오른쪽 서브트리 처리
            right_start, right_end = dfs(node.right)
            if right_start is not None:
                transitions.append([start, 'ε', right_start])
                transitions.append([right_end, 'ε', end])
                
            return start, end
            
        elif node.value == '*':  # 클레이니 스타
            start = create_state()
            end = create_state()
            
            # 서브트리 처리
            sub_start, sub_end = dfs(node.left)
            if sub_start is not None:
                transitions.append([start, 'ε', sub_start])
                transitions.append([sub_end, 'ε', sub_start])
                transitions.append([sub_end, 'ε', end])
                transitions.append([start, 'ε', end])
                
            return start, end
            
        elif node.value == '.':  # 접속
            left_start, left_end = dfs(node.left)
            right_start, right_end = dfs(node.right)
            
            if left_start is not None and right_start is not None:
                transitions.append([left_end, 'ε', right_start])
                return left_start, right_end
            return None, None
    
    # DFS로 트리 순회하며 e-NFA 구성
    start_state, final_state = dfs(tree)
    if final_state is not None:
        final_states.add(final_state)
    
    # 전이 함수를 2차원 배열로 변환
    transition_matrix = [[[] for _ in range(len(states))] for _ in range(len(states))]
    for transition in transitions:
        from_state, symbol, to_state = transition
        if symbol not in transition_matrix[from_state][to_state]:  # 중복 방지
            transition_matrix[from_state][to_state].append(symbol)
    
    return {
        'states': sorted(list(states)),
        'alphabet': sorted(list(alphabet)),
        'start_state': start_state,
        'transitions': transition_matrix,
        'final_states': sorted(list(final_states))
    }

if __name__ == "__main__":
    tree = RE2Tree.build_tree_from_re("aA+b+0c*")

    #e-NFA 생성
    eNFA = tree_to_eNFA(tree)

    #e-NFA 출력
    print("상태:")
    for state in sorted(eNFA['states']):
        print(f"상태 {state}")
    
    print("\n전이 함수:")
    for i in range(len(eNFA['transitions'])):
        for j in range(len(eNFA['transitions'][i])):
            if eNFA['transitions'][i][j]:
                print(f"{i} -> {eNFA['transitions'][i][j]} -> {j}")
    
    print(f"\n시작 상태: {eNFA['start_state']}")
    print(f"종결 상태: {eNFA['final_states']}")

    # 그래프와 표를 나란히 표시하기 위한 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가
    for state in eNFA['states']:
        G.add_node(state)
    
    # 엣지 추가
    for i in range(len(eNFA['transitions'])):
        for j in range(len(eNFA['transitions'][i])):
            if eNFA['transitions'][i][j]:
                G.add_edge(i, j)
    
    # 계층적 레이아웃 생성
    pos = {}
    
    # 시작 상태를 가장 왼쪽에 배치
    pos[eNFA['start_state']] = (0, 0)
    
    # 시작 상태로부터의 거리에 따라 x 좌표 조정
    distances = nx.single_source_shortest_path_length(G, eNFA['start_state'])
    max_distance = max(distances.values())
    
    # 각 거리별로 노드들을 수직으로 정렬
    nodes_by_distance = {}
    for node, distance in distances.items():
        if distance not in nodes_by_distance:
            nodes_by_distance[distance] = []
        nodes_by_distance[distance].append(node)
    
    # 종결 상태를 가장 오른쪽에 배치
    for final_state in eNFA['final_states']:
        pos[final_state] = (max_distance + 1, 0)
    
    # 각 거리별로 노드들을 수직으로 배치
    for distance in range(1, max_distance + 1):
        nodes = nodes_by_distance.get(distance, [])
        # 종결 상태는 제외
        nodes = [node for node in nodes if node not in eNFA['final_states']]
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
                          nodelist=[eNFA['start_state']],
                          node_color='green',
                          node_size=1000,
                          alpha=0.7,
                          ax=ax1)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=eNFA['final_states'],
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
        for j in range(len(eNFA['transitions'][state])):
            if eNFA['transitions'][state][j]:
                symbols.update(eNFA['transitions'][state][j])
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
    for i in range(len(eNFA['transitions'])):
        for j in range(len(eNFA['transitions'][i])):
            if eNFA['transitions'][i][j]:
                edge_labels[(i, j)] = ','.join(eNFA['transitions'][i][j])
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax1)
    
    ax1.set_title("e-NFA 그래프")
    ax1.axis('off')
    
    # 전이 함수 표 생성
    states = sorted(eNFA['states'])
    alphabet = sorted(eNFA['alphabet'])
    alphabet.append('ε')  # epsilon 전이를 위해 추가
    
    # 전이 함수 표 생성
    transition_table = []
    for state in states:
        row = []
        # 현재 상태의 심볼 정보 수집
        state_symbols = set()
        for j in range(len(eNFA['transitions'][state])):
            if eNFA['transitions'][state][j]:
                state_symbols.update(eNFA['transitions'][state][j])
        
        # 상태 레이블 생성
        state_label = f"q{state}"
        if state_symbols:
            state_label += f"\n({','.join(sorted(state_symbols))})"
            
        for symbol in alphabet:
            # 현재 상태에서 해당 심볼로 전이 가능한 모든 상태 찾기
            next_states = []
            for next_state in states:
                if eNFA['transitions'][state][next_state] and symbol in eNFA['transitions'][state][next_state]:
                    next_states.append(f"q{next_state}")
            row.append(','.join(next_states) if next_states else '')
        transition_table.append(row)
    
    # pandas DataFrame 상태 레이블: q0, q1, ... 형식
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
    
    # 시작 상태와 종결 상태 강조
    for i, state in enumerate(states):
        if state == eNFA['start_state']:
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('lightgreen')
        if state in eNFA['final_states']:
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('lightcoral')
    
    ax2.set_title("e-NFA 전이 함수 표")
    
    plt.tight_layout()
    plt.show()