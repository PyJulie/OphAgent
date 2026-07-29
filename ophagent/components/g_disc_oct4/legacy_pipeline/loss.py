# 必要に応じて損失関数を実装。
# main.py では参照していないが import エラー防止のため置いておく。

import torch.nn as nn

class DummyLoss(nn.Module):
    def forward(self, *args, **kwargs):
        return 0.
