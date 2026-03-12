import argparse
import pandas as pd
import os
import torch
from tree_sitter import Language, Parser, Node
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from tree import ASTNode, SingleNode


def parse_options():
    parser = argparse.ArgumentParser(description='TrVD preprocess.')
    parser.add_argument('-i', '--input',  default='mutrvd', type=str)
    parser.add_argument('-d', '--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str)
    return parser.parse_args()


args   = parse_options()
device = torch.device(args.device)

CPP_LANGUAGE = Language('build_languages/my-languages.so', 'cpp')
PARSER       = Parser()
PARSER.set_language(CPP_LANGUAGE)


SKIP_TYPES = {'{', '}', ';', '(', ')', ',', '\n', 'comment'}


class Pipeline:
    def __init__(self):
        self.train = self.val = self.test = None
        self.w2v_path = None

    ─────────────────────────────────────────────────────────
    @staticmethod
    def _node_stats(node):
        """ (max_depth, total_count)， DFS。"""
        stack     = [(node, 1)]
        max_depth = 0
        total     = 0
        while stack:
            cur, d = stack.pop()
            max_depth = max(max_depth, d)
            total    += 1
            for child in cur.children:
                stack.append((child, d + 1))
        return max_depth, total

    def controlled_subtree_decomposition(self, ast: Node, max_depth=8, max_nodes=40):
        key_types = {
            'function_declarator', 'if_statement',
            'for_statement',       'while_statement'
        }
        subtrees = []

        def decompose(node, current_depth=1):
            if not node or current_depth > max_depth:
                return
            depth, count = self._node_stats(node)

            if node.type in key_types and depth <= max_depth and count <= max_nodes:
             
                subtrees.append(node)

            elif node.type in key_types:
               
                for child in node.children:
                    if child.type == 'compound_statement':
                        decompose(child, current_depth + 1)

            else:
                if node.type not in SKIP_TYPES:
                    if node.children:
                      
                        subtrees.append(node)
                   
                else:
                    
                    for child in node.children:
                        decompose(child, current_depth + 1)

        decompose(ast.root_node)
        return subtrees

    def build_syntax_based_cfg(self, subtree: Node, max_depth=30):
        G               = nx.DiGraph()
        node_id_counter = [0]

        def is_statement_node(n):
            excluded = {
                'break_statement', 'continue_statement',
                'return_statement', 'throw_statement'
            }
            if n.type in excluded:
                return False
            if n.type == 'expression_statement':
                return any(c.type in {
                    'call_expression', 'assignment_expression',
                    'binary_expression', 'unary_expression'
                } for c in n.children)
            return n.type in {
                'if_statement', 'for_statement', 'while_statement',
                'compound_statement', 'switch_statement',
                'case_statement',     'default_statement'
            }

        def add_node(n):
            nid = node_id_counter[0]
            node_id_counter[0] += 1
            G.add_node(
                nid,
                type=n.type,
                text=n.text.decode('utf-8') if n.text else n.type
            )
            return nid

        def get_next_statement(n, parent_id=None, depth=0):
            if not n or depth > max_depth:
                return None
            if n.type in ('{', '}', ';'):
                return None

            curr_id  = add_node(n)
            children = n.children
            if parent_id is not None:
                G.add_edge(parent_id, curr_id)

            if n.type == 'if_statement':
                # tree-sitter C++ grammar：条件节点类型为 condition_clause
                condition   = next((c for c in children if c.type == 'condition_clause'), None)
                then_branch = next((c for c in children if c.type == 'compound_statement'), None)
                else_branch = next((c for c in children if c.type == 'else_clause'), None)
                cond_id = get_next_statement(condition, curr_id, depth + 1) if condition else curr_id
                if then_branch:
                    get_next_statement(then_branch, cond_id, depth + 1)
                if else_branch:
                    get_next_statement(else_branch, cond_id, depth + 1)

            elif n.type == 'while_statement':
             
                condition = next((c for c in children if c.type == 'condition_clause'), None)
                body      = next((c for c in children if c.type == 'compound_statement'), None)
                cond_id   = get_next_statement(condition, curr_id, depth + 1) if condition else curr_id
                if body:
                    body_exit = get_next_statement(body, cond_id, depth + 1)
                    if body_exit is not None:
                        G.add_edge(body_exit, cond_id)  # 循环回边

            elif n.type == 'for_statement':
               
                non_punct = [c for c in children if c.type not in (';', '(', ')')]
                condition = non_punct[1] if len(non_punct) > 1 else None
                body      = next((c for c in children if c.type == 'compound_statement'), None)
                cond_id   = get_next_statement(condition, curr_id, depth + 1) if condition else curr_id
                if body:
                    body_exit = get_next_statement(body, cond_id, depth + 1)
                    if body_exit is not None:
                        G.add_edge(body_exit, cond_id)  # 循环回边

            elif n.type == 'switch_statement':
                # tree-sitter C++ grammar：条件节点类型为 condition_clause
                condition = next((c for c in children if c.type == 'condition_clause'), None)
                body      = next((c for c in children if c.type == 'compound_statement'), None)
                cond_id   = get_next_statement(condition, curr_id, depth + 1) if condition else curr_id
                if body:
                    for child in body.children:
                        if child.type in {'case_statement', 'default_statement'}:
                            get_next_statement(child, cond_id, depth + 1)

            else:
               
                prev_id = curr_id
                for child in children:
                    if is_statement_node(child):
                        child_id = get_next_statement(child, prev_id, depth + 1)
                        if child_id is not None:
                            prev_id = child_id
                    else:
                        get_next_statement(child, curr_id, depth + 1)

            return curr_id

        get_next_statement(subtree)
        if not G.nodes:
            G.add_node(0, type='root', text='root')
        return G

    def select_paths(self, G, entry_node=0, m=3):
       
        if entry_node not in G:
            return []

        V  = list(G.nodes)
        Vb = {n for n in V if any(
            k in G.nodes[n].get('type', '') for k in ('if', 'for', 'while')
        )}
        w = {e: 1.0 for e in G.edges}

        def weight_fn(u, v, d):
            return w.get((u, v), 1.0)

        V_uc = set(V)
        P    = []

        for _ in range(m):
            try:
                _, paths = nx.single_source_dijkstra(G, entry_node, weight=weight_fn)
            except Exception:
                break

            best_path, max_uncovered = None, -1
            for path in paths.values():
                uncovered = sum(1 for n in path if n in V_uc)
                if uncovered > max_uncovered:
                    max_uncovered = uncovered
                    best_path     = path

            if not best_path:
                break

            P.append(best_path)
            V_uc -= set(best_path)

            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                if u in Vb and (u, v) in w:
                    w[(u, v)] += 100.0

        return P[:m]

    def generate_dfg_sequence(self, subtree: Node, vocab: dict, max_token: int):
        
        variables       = {}
        dfg_seq_indices = []

        def register_var(name, def_type):
            name = name.lstrip('*& ')
            if name:
                variables[name] = def_type

        def traverse(n):
            if not n:
                return
            if n.type == 'declaration':
                d = n.child_by_field_name('declarator')
                if d:
                    register_var(d.text.decode('utf-8').strip(), 'declaration')
            elif n.type == 'parameter_declaration':
                d = n.child_by_field_name('declarator')
                if d:
                    register_var(d.text.decode('utf-8').strip(), 'parameter')
            elif n.type == 'assignment_expression':
                lhs = n.child_by_field_name('left')
                if lhs and lhs.type == 'identifier':
                    vn = lhs.text.decode('utf-8').strip()
                    if vn not in variables:
                        register_var(vn, 'assignment')
            elif n.type == 'identifier':
                vn = n.text.decode('utf-8').strip()
                if vn in variables:
                    dfg_seq_indices.extend([
                        vocab.get(variables[vn], max_token),
                        vocab.get('identifier',  max_token)
                    ])
            for child in n.children:
                traverse(child)

        traverse(subtree)
        return dfg_seq_indices if dfg_seq_indices else [max_token]

    def parse_source(self, dataset):
        for split in ['train', 'val', 'test']:
            try:
                print(f"Loading {split}.pkl...")
                data = pd.read_pickle(f'./{dataset}/{split}.pkl')
                if data.empty:
                    raise ValueError(f"{split}.pkl is empty.")

                rows = [row for _, row in data.iterrows()]

                def parse_one(row):
                    return self.controlled_subtree_decomposition(
                        PARSER.parse(row['code'].encode())
                    )

                # executor.map 保序 + tqdm 惰性迭代，进度条语义正确
                with ThreadPoolExecutor(max_workers=8) as executor:
                    subtrees_list = list(tqdm(
                        executor.map(parse_one, rows),
                        total=len(rows),
                        desc=f"Parsing {split} subtrees"
                    ))

                data             = data.copy()
                data['subtrees'] = subtrees_list
                setattr(self, split, data)

            except Exception as e:
                print(f"Error processing {split}: {str(e)}")
                continue

    def dictionary_and_embedding(self, vector_size=128):
        self.w2v_path = f'subtrees/{args.input}/node_w2v_{vector_size}'
        os.makedirs(f'subtrees/{args.input}', exist_ok=True)

        corpus = []
        for subtree_list in tqdm(self.train['subtrees'], desc="Generating corpus"):
            for st in subtree_list:
                st_node = ASTNode(st)
               
                tokens = []
                stack  = [st_node]
                while stack:
                    cur = stack.pop()
                    if cur.token:
                        tokens.append(cur.token)
                    stack.extend(cur.children)
                if tokens:
                    corpus.append(tokens)

        w2v = Word2Vec(corpus, vector_size=vector_size, workers=8, sg=1, min_count=3)
        w2v.save(self.w2v_path)

    def tree_to_index(self, node, vocab, max_token):
        ast_node = ASTNode(node) if isinstance(node, Node) else node
        token    = ast_node.token
        result   = [vocab.get(token, max_token)]
        children = [self.tree_to_index(c, vocab, max_token) for c in ast_node.children]
        if children:
            result.append(children)
        return result

    def generate_features(self, data, name: str):
        if data is None:
            return

        w2v       = Word2Vec.load(self.w2v_path).wv
        vocab     = w2v.key_to_index
        max_token = w2v.vectors.shape[0]

        def process_row(row):
            # 1. SFE
            subtrees = [self.tree_to_index(st, vocab, max_token) for st in row['subtrees']]

            # 2. CFPE
            if row['subtrees']:
                cfg_G     = self.build_syntax_based_cfg(row['subtrees'][0])
                paths_ids = self.select_paths(cfg_G, m=3)
            else:
                cfg_G     = nx.DiGraph()
                cfg_G.add_node(0, type='root', text='root')
                paths_ids = []

            cfg_paths = []
            for path in paths_ids:
                path_idx = [
                    vocab.get(cfg_G.nodes[n].get('type', 'unknown'), max_token)
                    for n in path if n in cfg_G.nodes
                ]
                if path_idx:
                    cfg_paths.append(path_idx)

            
            while len(cfg_paths) < 3:
                cfg_paths.append([max_token])

            
            if row['subtrees']:
                dfg_seqs = []
                for st in row['subtrees']:
                    dfg_seqs.extend(self.generate_dfg_sequence(st, vocab, max_token))
                dfg_seqs = dfg_seqs if dfg_seqs else [max_token]
            else:
                dfg_seqs = [max_token]

            return {'subtrees': subtrees, 'cfg_paths': cfg_paths, 'dfg_seqs': dfg_seqs}

        rows = [row for _, row in data.iterrows()]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(
                executor.map(process_row, rows),
                total=len(rows),
                desc=f"Generating {name} features"
            ))

        out             = data.copy()
        out['features'] = results
        out             = out.drop(columns=['subtrees'])
        out.to_pickle(f'subtrees/{args.input}/{name}_features.pkl')

    def run(self, dataset):
        self.parse_source(dataset)
        self.dictionary_and_embedding()
        self.generate_features(self.train, 'train')
        self.generate_features(self.val,   'val')
        self.generate_features(self.test,  'test')


if __name__ == '__main__':
    ppl = Pipeline()
    ppl.run(args.input)