import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def matrix_mul(input_tensor, weight, bias=None):
   
    out = torch.matmul(input_tensor, weight)
    if bias is not None:
        out = out + bias
    return torch.tanh(out)


def element_wise_mul(input1, input2):
   
    weight   = input2.unsqueeze(-1).expand_as(input1)
    weighted = (input1 * weight).sum(dim=0, keepdim=True)
    return weighted


class BatchTreeEncoder(nn.Module):
    

    def __init__(self, vocab_size, embedding_dim, encode_dim,
                 batch_size, use_gpu, device, pretrained_weight=None):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim   = encode_dim
        self.batch_size   = batch_size
        self.use_gpu      = use_gpu
        self.device       = device
        self.node_list    = []
        self.batch_node   = None

        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.agg_net        = nn.GRU(embedding_dim, encode_dim, 1)
        self.sent_weight    = nn.Parameter(torch.Tensor(encode_dim, encode_dim))
        self.sent_bias      = nn.Parameter(torch.Tensor(1, encode_dim))
        self.context_weight = nn.Parameter(torch.Tensor(encode_dim, 1))
        self.use_att        = True
        self._init_weights()

    def _init_weights(self, mean=0.0, std=0.05):
        for p in (self.sent_weight, self.context_weight, self.sent_bias):
            p.data.normal_(mean, std)

    def create_tensor(self, tensor):
        return tensor.to(self.device)

    def traverse_mul(self, node, batch_index):
       
        batch_index = list(batch_index)

        size = len(node)
        if not size:
            return None

        
        batch_current = self.create_tensor(torch.zeros(size, self.embedding_dim))
        index, children_index, current_node, children = [], [], [], []

        for i in range(size):
            if isinstance(node[i], list) and node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                if len(node[i]) > 1 and isinstance(node[i][1], list):
                    children.append(node[i][1])
                   
                    children_index.append([i] * len(node[i][1]))
            else:
                batch_index[i] = -1

        if not index:
            return None

        batch_current = batch_current.index_copy(
            0,
            torch.tensor(index, dtype=torch.long, device=self.device),
            self.embedding(torch.tensor(current_node, dtype=torch.long, device=self.device))
        )

        childs_hidden_sum = self.create_tensor(torch.zeros(size, self.encode_dim))
        hidden_per_child  = []

        for c in range(len(children)):
            zeros              = self.create_tensor(torch.zeros(size, self.encode_dim))
            batch_children_idx = [batch_index[i] for i in children_index[c]]
            tree               = self.traverse_mul(children[c], batch_children_idx)
            if tree is not None:
                cur_child = zeros.index_copy(
                    0,
                    torch.tensor(children_index[c], dtype=torch.long, device=self.device),
                    tree
                )
                childs_hidden_sum += cur_child
                hidden_per_child.append(cur_child)

        if self.use_att and hidden_per_child:
            child_hiddens   = torch.stack(hidden_per_child)              # (K, batch, dim)
            childs_weighted = matrix_mul(child_hiddens, self.sent_weight, self.sent_bias)
            childs_weighted = matrix_mul(childs_weighted, self.context_weight)  # (K, batch, 1)
            childs_weighted = childs_weighted.squeeze(-1).permute(1, 0)         # (batch, K)
            childs_weighted = F.softmax(childs_weighted, dim=-1).permute(1, 0)  # (K, batch)
            childs_hidden_sum = element_wise_mul(child_hiddens, childs_weighted).squeeze(0)

        batch_current     = batch_current.unsqueeze(0)
        childs_hidden_sum = childs_hidden_sum.unsqueeze(0)
        _, hn             = self.agg_net(batch_current, childs_hidden_sum)
        hn                = hn.squeeze(0)   # (batch, encode_dim)

        valid_indices = [i for i in batch_index if i != -1]
        if valid_indices:
            b_in   = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
            nd_tmp = self.batch_node.index_copy(0, b_in, hn)
            self.node_list.append(nd_tmp)

        return hn

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(torch.zeros(bs, self.encode_dim))
        self.node_list  = []
        self.traverse_mul(x, list(range(bs)))

        if not self.node_list:
            return self.create_tensor(torch.zeros(bs, self.encode_dim))

        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, dim=0)[0]


class PositionalEncoding(nn.Module):
    
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe   = torch.zeros(max_len, embed_dim)
        pos  = torch.arange(0, max_len).unsqueeze(1).float()
        half = embed_dim // 2
        div  = torch.exp(
            torch.arange(0, half).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div[:half])
        pe[:, 1::2] = torch.cos(pos * div[:embed_dim - half])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CFPE(nn.Module):
    """Control Flow Path Extractor (Transformer + Positional Encoding)"""

    def __init__(self, vocab_size, embed_dim, device):
        super().__init__()
        self.device      = device
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc     = PositionalEncoding(embed_dim)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, cfg_paths):
        batch_size, m_paths, seq_len = cfg_paths.size()
        paths_flat = cfg_paths.view(batch_size * m_paths, seq_len)
        embedded   = self.embedding(paths_flat)
        embedded   = self.pos_enc(embedded)
        encoded    = self.transformer(embedded)
        h_path     = encoded.mean(dim=1)
        return h_path.view(batch_size, m_paths, -1)


class DFFE(nn.Module):
    """Data Flow Feature Extractor (GRU)"""

    def __init__(self, vocab_size, embed_dim, device):
        super().__init__()
        self.device    = device
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.gru     = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, dfg_seq):
        embedded = self.embedding(dfg_seq)
        out, _   = self.gru(embedded)
        h_dfg    = self.dropout(out.mean(dim=1))
        return h_dfg


class BatchProgramClassifier(nn.Module):
    """Multi-Level Feature Fusion Module (MLFFM)"""

    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim,
                 label_size, batch_size, device, use_gpu=True, pretrained_weight=None):
        super().__init__()
        self.batch_size = batch_size
        self.encode_dim = encode_dim
        self.device     = device

        self.sfe  = BatchTreeEncoder(
            vocab_size, embedding_dim, encode_dim,
            batch_size, use_gpu, device, pretrained_weight
        )
        self.cfpe = CFPE(vocab_size, encode_dim, device)
        self.dffe = DFFE(vocab_size, encode_dim, device)

        
        self.cfg_fusion = nn.Linear(encode_dim * 3, encode_dim)
        self.linear_dfg = nn.Linear(encode_dim, encode_dim)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=encode_dim, nhead=4, batch_first=True, dropout=0.1
        )
        self.global_transformer = nn.TransformerEncoder(fusion_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(encode_dim * 3),
            nn.Dropout(0.3),
            nn.Linear(encode_dim * 3, encode_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encode_dim, label_size)
        )

    def forward(self, subtrees, cfg_paths, dfg_seqs):
        real_bs = len(subtrees)

       
        filter_tree = [[st for st in tree if len(st) > 1] for tree in subtrees]
        lens        = [len(item) for item in filter_tree]
        encodes     = [st for tree in filter_tree for st in tree]
        total_nodes = sum(lens)

        if total_nodes > 0:
            tree_encodes = self.sfe(encodes, total_nodes)

            if tree_encodes.size(0) == 0:
                h_subtree = torch.zeros(real_bs, self.encode_dim, device=self.device)
            else:
                max_len    = max(lens)
                seq, start = [], 0
                for l in lens:
                    if l > 0:
                        chunk = tree_encodes[start:start + l]
                        pad   = torch.zeros(max_len - l, self.encode_dim, device=self.device)
                        seq.append(torch.cat([chunk, pad], dim=0))
                    else:
                        seq.append(torch.zeros(max_len, self.encode_dim, device=self.device))
                    start += l

                ast_seq = torch.stack(seq)  # (B, max_len, dim)

               
                mask = torch.zeros(real_bs, max_len, dtype=torch.bool, device=self.device)
                for i, l in enumerate(lens):
                    if l > 0:
                        mask[i, :l] = True

                ast_seq_masked = ast_seq.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                h_subtree      = torch.max(ast_seq_masked, dim=1)[0]
               
                h_subtree      = h_subtree.masked_fill(h_subtree == float('-inf'), 0.0)
        else:
            h_subtree = torch.zeros(real_bs, self.encode_dim, device=self.device)

        H_cfg  = self.cfpe(cfg_paths)   # (B, 3, dim)
        h_dfg  = self.dffe(dfg_seqs)    # (B, dim)

   
        h_cfgfused = self.cfg_fusion(H_cfg.view(real_bs, -1))  # (B, dim)
        h_dfgfused = self.linear_dfg(h_dfg)                    # (B, dim)

        h_combined = torch.stack([h_subtree, h_cfgfused, h_dfgfused], dim=1)  # (B, 3, dim)
        h_global   = self.global_transformer(h_combined)                       # (B, 3, dim)

   
        h_global_flat = h_global.view(h_global.size(0), -1)   # (B, 3*dim)
        logits        = self.classifier(h_global_flat)
        return logits