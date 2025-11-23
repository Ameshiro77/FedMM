import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
from utils.model_utils import *
from FLAlgorithms.config import EVAL_WIN

class User:
    def __init__(self, client_id, train_data, test_data, public_data,
                 ae_model, cf_model, modalities,
                 batch_size=32, learning_rate=1e-3,
                 beta=1.0, lamda=0.0, local_epochs=1,
                 label_ratio=0.1, dataset_name="opp"):

        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.modalities = modalities if isinstance(modalities, list) else [modalities]

        self.train_data = train_data
        self.test_data = test_data
        self.public_data = public_data

        n_samples = len(train_data["y"])
        n_labeled = max(1, int(label_ratio * n_samples))

        # 连续取最后的 n_labeled 数据作为有标签
        labeled_idx = np.arange(n_samples - n_labeled, n_samples)
        unlabeled_idx = np.arange(0, n_samples - n_labeled)

        self.labeled_data = {
            m: train_data[m][labeled_idx] for m in self.modalities
        }
        self.labeled_data["y"] = train_data["y"][labeled_idx]

        self.unlabeled_data = {
            m: train_data[m][unlabeled_idx] for m in self.modalities
        }
        self.unlabeled_data["y"] = train_data["y"][unlabeled_idx]

        self.train_samples = len(self.unlabeled_data["y"])
        self.ae_model = copy.deepcopy(ae_model).to(self.device)
        self.cf_model = copy.deepcopy(cf_model).to(self.device)
        self.rec_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(
        #     list(self.ae_model.parameters()) + list(self.cf_model.parameters()),
        #     lr=self.learning_rate
        # )
        self.optimizer_ae = torch.optim.Adam(self.ae_model.parameters(), lr=self.learning_rate)
        self.optimizer_cf = torch.optim.Adam(self.cf_model.parameters(), lr=self.learning_rate)

    def info(self):
        print(f"\n=== Client {self.client_id} ===")
        print(f"Modalities: {self.modalities}")
        print(f"Labeled samples: {len(self.labeled_data['y'])}")
        print(f"Unlabeled samples: {len(self.unlabeled_data['y'])}")
        print(f"Total train samples: {len(self.train_data['y'])}")

        # 分别打印三个部分的分布
        print_dataset_info(self.train_data, name=f"Client {self.client_id} - Train Data")
        print_dataset_info(self.labeled_data, name=f"Client {self.client_id} - Labeled Subset")
        print_dataset_info(self.unlabeled_data, name=f"Client {self.client_id} - Unlabeled Subset")
        

    def freeze(self, sub_model):
        """Freeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = False

    def unfreeze(self, sub_model):
        """Unfreeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = True
    def set_ae_parameters(self, ae_state_dicts):
        current_state_dict = self.ae_model.state_dict()
        
        updated_state_dict = {}
        for key in current_state_dict.keys():
            if key in ae_state_dicts:
                updated_state_dict[key] = ae_state_dicts[key]
            else:
                updated_state_dict[key] = current_state_dict[key]
        
        self.ae_model.load_state_dict(updated_state_dict)

    def get_ae_parameters(self):
        # 返回整个 AE 的 state_dict
        return self.ae_model.state_dict()

    def get_grads(self):
        grads = []
        for m in self.modalities:
            for p in self.ae_model.encoders[m].parameters():
                grads.append(p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for p in self.ae_model.decoders[m].parameters():
                grads.append(p.grad.clone() if p.grad is not None else torch.zeros_like(p))
        for p in self.cf_model.parameters():
            grads.append(p.grad.clone() if p.grad is not None else torch.zeros_like(p))
        return grads

    def update_parameters(self, new_params):
        ae_params, cf_params = new_params
        self.set_parameters(ae_params, cf_params)

    def train_ae(self):
        self.ae_model.train()
        self.cf_model.eval()

        total_loss = 0.0
        for ep in range(self.local_epochs):
            modalities_seq, _ = make_seq_batch2(self.unlabeled_data, self.batch_size)
            X_modal = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_modal[self.modalities[0]].shape[1]

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer_ae.zero_grad()
                rec_loss = 0.0

                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    recon = self.ae_model(x_m, m)
                    rec_loss += self.rec_loss_fn(recon, x_m)

                rec_loss.backward()
                self.optimizer_ae.step()
                total_loss += rec_loss.item()

        return total_loss / max(1, seq_len // self.batch_size)

    def train_cf(self):
        self.freeze(self.ae_model)
        self.cf_model.train()

        for epoch in range(self.local_epochs):
            total_loss, total_correct, total_samples = 0.0, 0, 0
            
            # 构造有监督数据序列
            # print(self.batch_size,len(self.labeled_data['y']))
            modalities_seq, y_seq = make_seq_batch2(self.labeled_data, self.batch_size)
            X_modal = {m: modalities_seq[m] for m in self.modalities}
            seq_len = X_modal[self.modalities[0]].shape[1]

            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(16, 32)
                idx_start = idx_end
                idx_end = min(idx_end + win_len, seq_len)

                self.optimizer_cf.zero_grad()
                latents = []

                # 取每个模态的 latent
                for m in self.modalities:
                    x_m = torch.from_numpy(X_modal[m][:, idx_start:idx_end, :]).to(self.device)
                    latents.append(self.ae_model.encode(x_m, m)[1])

                latent_cat = torch.cat(latents, dim=1)
                y_batch = torch.from_numpy(y_seq[:, idx_start:idx_end]).to(self.device).reshape(-1).long()

                logits = self.cf_model(latent_cat)
                loss = self.cls_loss_fn(logits, y_batch)

                loss.backward()
                self.optimizer_cf.step()

                total_loss += loss.item()
                total_correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
                total_samples += y_batch.size(0)

            # 每轮结束计算指标
            epoch_loss = total_loss / max(1, seq_len // self.batch_size)
            epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
            
            # 打印训练信息（可选）
            print(f'Epoch [{epoch+1}/{self.local_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        self.unfreeze(self.ae_model)
        return epoch_loss, epoch_acc

    # -----------------------------
    # 联合训练（重构 + 分类）
    # -----------------------------

    def train_all(self):
        self.ae_model.train()
        self.cf_model.train()

        total_loss, total_rec, total_cls = 0.0, 0.0, 0.0
        total_correct, total_samples = 0, 0

        iter_labeled = iter(self.trainloader_labeled)
        iters = len(self.trainloader_unlabeled)

        for batch_u in self.trainloader_unlabeled:
            *Xs_u, _ = batch_u
            Xs_u = [x.to(self.device) for x in Xs_u]
            self.optimizer.zero_grad()

            # ae
            rec_loss = 0.0
            for x_m, m in zip(Xs_u, self.modalities):
                recon = self.ae_model.forward(x_m, m)
                rec_loss += self.rec_loss_fn(recon, x_m)

            print("labeled dataset size:", len(self.trainloader_labeled.dataset))
            print("unlabeled dataset size:", len(self.trainloader_unlabeled.dataset))
            print("labeled batch size:", self.trainloader_labeled.batch_size)

            # cf; Dl < Du,loader need reset
            try:
                batch_l = next(iter_labeled)
            except StopIteration:
                iter_labeled = iter(self.trainloader_labeled)
                batch_l = next(iter_labeled)
            *Xs_l, y_l = batch_l
            Xs_l = [x.to(self.device) for x in Xs_l]
            y_l = y_l.to(self.device)

            latents_l = []
            for x_m, m in zip(Xs_l, self.modalities):
                latent = self.encode(x_m, m)
                latents_l.append(latent)

            latent_cat = torch.cat(latents_l, dim=1)
            logits = self.cf_model(latent_cat)
            cls_loss = self.cls_loss_fn(logits, y_l)

            # ----------------------
            # 总损失 & 更新
            # ----------------------
            loss = self.beta * rec_loss + cls_loss
            loss.backward()
            self.optimizer.step()

            # ----------------------
            # 统计
            # ----------------------
            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_cls += cls_loss.item()
            total_correct += (torch.argmax(logits, dim=1) == y_l).sum().item()
            total_samples += y_l.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        return total_loss / iters, total_rec / iters, total_cls / iters, acc


    def test(self, eval_win=EVAL_WIN):
        self.ae_model.eval()
        self.cf_model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for start in range(0, len(self.test_data["y"]) - eval_win + 1, eval_win):
                batch_x = {m: torch.tensor(
                    self.test_data[m][start:start + eval_win], dtype=torch.float32, device=self.device
                ) for m in self.modalities}
                batch_y = torch.tensor(
                    self.test_data["y"][start:start + eval_win], dtype=torch.long, device=self.device
                )
                # ------- encode -------
                reps = []
                for m in self.modalities:
                    _, out = self.ae_model.encode(batch_x[m], m)
                    reps.append(out)

                reps = torch.cat(reps, dim=-1)   # 拼接多个模态表示

                # ------- classifier -------
                outputs = self.cf_model(reps)
                # print(outputs.shape, batch_y.shape)
                loss = self.cls_loss_fn(outputs, batch_y)

                total_loss += loss.item() * batch_y.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += torch.sum(preds == batch_y).item()
                total_samples += batch_y.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_acc, None
