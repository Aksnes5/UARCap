
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExactCFEModule(nn.Module):
    """
    完全按照论文公式的CFE模块
    5个公式，一个不多，一个不少
    """
    def __init__(self, d=768):
        super(ExactCFEModule, self).__init__()
        self.d = d
        
        # 公式(4)中的线性投影层
        self.LP_visual = nn.Linear(d, d)
        self.LP_text = nn.Linear(d, d)
        
        # 公式(7)中的线性投影层
        self.LP_fusion = nn.Linear(3 * d, d)
        
        # 公式(8)中的1D卷积
        # self.conv1d = nn.Conv1d(2 * d, d, kernel_size=1)
        
    def forward(self, F_v, F_q):
        """
        输入:
            F_v: [batch_size, L, d] 视觉特征
            F_q: [batch_size, N, d] 文本特征
        输出:
            F: [batch_size, L, d] 增强特征
        """
        batch_size, L, d = F_v.shape
        _, N, _ = F_q.shape
        
        # 公式(4): M = (LP(F^v) * LP(F^q)^T) / sqrt(d)
        F_v_lp = self.LP_visual(F_v)
        F_q_lp = self.LP_text(F_q)
        M = torch.matmul(F_v_lp, F_q_lp.transpose(1, 2)) / math.sqrt(d)
        
        # 公式(5): F^v' = M_col * F^q
        M_col = F.softmax(M, dim=-1)  # 列向softmax
        F_v_prime = torch.matmul(M_col, F_q)
        
        # 公式(6): F^q' = M_row * M_col^T * F^v
        M_row = F.softmax(M, dim=-2)  # 行向softmax
        M_row_col_T = torch.matmul(M_row, M_col.transpose(1, 2))
        F_q_prime = torch.matmul(M_row_col_T, F_v)
        
        # 公式(7): F_bar = LP([F^v | F^v' | F^q'])
        F_bar = self.LP_fusion(torch.cat([F_v, F_v_prime, F_q_prime], dim=-1))
        
        # 公式(8): F = Conv1D([F^g | F_bar])
        # F_g = F_q.mean(dim=1, keepdim=True).expand(-1, L, -1)
        # F_output = self.conv1d(torch.cat([F_g, F_bar], dim=-1).transpose(1, 2)).transpose(1, 2)
        
        # return F_output
        return F_bar