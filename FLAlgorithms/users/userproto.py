

import numpy as np
import torch
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
from utils.model_utils import *

class UserProto(User):
    """
    FedProto baseline 客户端
    - 上传类原型
    - 使用 global_prototypes 做 proto alignment
    """

    def __init__(self, client_id, train_data, test_data, public_data,
                 ae_model, cf_model, modalities,
                 batch_size, learning_rate, beta, lamda,
                 local_epochs, label_ratio):

        super().__init__(client_id, train_data, test_data, public_data,
                         ae_model, cf_model, modalities,
                         batch_size, learning_rate, beta, lamda,
                         local_epochs, label_ratio)

    # ----------------------------------------------------
    # 上传 prototype （只使用 labeled data）
    # ----------------------------------------------------
    def upload_prototype(self):
        self.ae_model.eval()

        X_modal = {m: self.labeled_data[m] for m in self.modalities}
        y_all = self.labeled_data["y"]   # [N]
        y_all = np.array(y_all)

        classes = np.unique(y_all)
        feats_per_class = {int(c): [] for c in classes}

        # 整体 forward
        for m in self.modalities:
            x = torch.tensor(X_modal[m], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ , z = self.ae_model.encode(x, m)

            for cls in classes:
                mask = (y_all == cls)
                if mask.sum() > 0:
                    feats_per_class[int(cls)].append(z[mask])

        # === 聚合每类 ===
        prototypes = {}
        proto_weights = {}
        for cls, feat_list in feats_per_class.items():
            if len(feat_list) == 0:
                continue
            z_all = torch.cat(feat_list, dim=0)
            prototypes[cls] = z_all.mean(dim=0).detach().cpu()
            proto_weights[cls] = z_all.shape[0]

        return prototypes, proto_weights

    # ----------------------------------------------------
    # 使用 server 下发的 global prototypes 做分类器训练
    # ----------------------------------------------------
    def train_cf_proto(self, global_prototypes, beta_proto=1.0):

        self.ae_model.eval()
        self.cf_model.train()

        loss_list = []

        modalities_seq, y_seq = make_seq_batch2(self.labeled_data, self.batch_size)
        X_modal = {m: modalities_seq[m] for m in self.modalities}
        seq_len = X_modal[self.modalities[0]].shape[1]

        for ep in range(self.local_epochs):
            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start, idx_end = idx_end, min(idx_end + win_len, seq_len)

                self.optimizer_cf.zero_grad()

                # multi-modal encode
                latents = []
                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end]).to(self.device)
                    _, z = self.ae_model.encode(x_m, m)
                    latents.append(z)

                z_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).reshape(-1).to(self.device)

                # classifier
                logits = self.cf_model(z_cat)
                ce_loss = self.cls_loss_fn(logits, y_batch)

                # ===== Proto Loss =====
                proto_loss = 0.0
                if global_prototypes is not None:
                    z_flat = z_cat.view(-1, z_cat.shape[-1])
                    y_flat = y_batch.view(-1)

                    for cls, proto_g in global_prototypes.items():
                        mask = (y_flat == cls)
                        if mask.sum() > 0:
                            z_cls = z_flat[mask]
                            proto_loss += F.mse_loss(z_cls.mean(dim=0),
                                                     proto_g.to(self.device))

                loss = ce_loss + beta_proto * proto_loss
                loss.backward()
                self.optimizer_cf.step()

                loss_list.append(loss.item())

        return {"cf_loss": np.mean(loss_list)}
