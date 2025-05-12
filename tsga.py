import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import hydra
import time
from collections import OrderedDict
from utils import set_device, set_seed, time_monitor, save_statistics, OCCData, GradientDesent, LossNet, AdaptNet


class Experiment(nn.Module):
    def __init__(self, args, device):
        super(Experiment, self).__init__()
        self.args = args
        self.device = device
        self.data = OCCData(self.args)
        fc_in = self.args.num_filters * (self.args.image_height // 16) * (self.args.image_width // 16)
        if self.args.backbone == 'ResNet12':
            if self.args.anil:
                from resnet_anil import ResNet12
            else:
                from resnet import ResNet12
            self.model = ResNet12(self.args.image_channels, [16, 32, 64, 128]).to(self.device)  # [64, 128, 256, 512]
        elif self.args.backbone == 'CONV4':
            if self.args.anil:
                from model_anil import CONV4
            else:
                from model import CONV4
            self.model = CONV4(self.args.image_channels, self.args.num_filters, fc_in).to(self.device)
            if self.args.pretrain:
                print('load pretrain weights...')
                self.model.load_state_dict(torch.load('/data/lyl/code/simple_MAML/conv-4_pretrained_3.0.pth', map_location=self.device), strict=False)
        else:
            print('Wrong backbone !')
            exit()
        print('model: {:.2f}'.format(sum(p.numel() for p in self.model.parameters())))

        self.gd = GradientDesent(
            OrderedDict(self.model.meta_named_parameters()), 
            self.args.n_inner_steps, self.args.mamlpp, self.args.inner_lr, wd=self.args.weight_decay
        ).to(self.device)
        print('gd: {:.2f}'.format(sum(p.numel() for p in self.gd.parameters() if p.requires_grad)))

        if self.args.loss:
            self.loss = LossNet(self.args).to(self.device)
            print('loss: {:.2f}'.format(sum(p.numel() for p in self.loss.parameters())))

        if self.args.adapter:
            self.adapter = AdaptNet(self.args).to(self.device)
            print('adapter: {:.2f}'.format(sum(p.numel() for p in self.adapter.parameters())))

        self.optimizer = optim.AdamW(self.parameters(), lr=self.args.outer_lr)

    def adaptation(self, s_x, s_y, q_x):
        params = None
        for step in range(self.args.n_inner_steps):
            s_p, s_e = self.model(s_x, params=params)
            if self.args.loss:
                inner_loss = self.loss(s_p)
            else:
                inner_loss = F.cross_entropy(s_p, s_y)
            if self.args.adapter:
                params = self.gd.update(self.model, inner_loss, step, params=params, adapter=self.adapter)
            else:
                params = self.gd.update(self.model, inner_loss, step, params=params)
        q_p, q_e = self.model(q_x, params)
        return q_p
    
    def train(self, n_tasks, get_episode):
        tic = time.time()
        accs = []
        for _ in range(n_tasks):
            self.model.zero_grad()
            support_X, support_y, query_X, query_y = get_episode()
            support_X, support_y, query_X, query_y = support_X.to(self.device), support_y.to(self.device), query_X.to(self.device), query_y.to(self.device)
            outer_loss = torch.tensor(0., device=self.device)
            for task_idx, (s_x, s_y, q_x, q_y) in enumerate(zip(support_X, support_y, query_X, query_y)):
                self.model.zero_grad()
                preds = self.adaptation(s_x, s_y, q_x)
                with torch.no_grad():
                    _, p = torch.max(preds, dim=-1)
                    accs.append(torch.mean(p.eq(q_y).float()))  # accuracy of a task
                outer_loss += F.cross_entropy(preds, q_y)
            outer_loss.div_(self.args.batch_size)

            outer_loss.backward()
            clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
        accs = torch.Tensor(accs)
        toc = time.time()
        return accs.mean() * 100, accs.std() * 100, toc - tic

    def evaluate(self, n_tasks, get_episode):
        tic = time.time()
        accs = []
        for _ in range(n_tasks):
            support_X, support_y, query_X, query_y = get_episode()
            support_X, support_y, query_X, query_y = support_X.to(self.device), support_y.to(self.device), query_X.to(self.device), query_y.to(self.device)
            for task_idx, (s_x, s_y, q_x, q_y) in enumerate(zip(support_X, support_y, query_X, query_y)):
                self.model.zero_grad()
                preds = self.adaptation(s_x, s_y, q_x)
                with torch.no_grad():
                    _, p = torch.max(preds, dim=-1)
                    accs.append(torch.mean(p.eq(q_y).float()))  # accuracy of a task
        accs = torch.Tensor(accs)
        toc = time.time()
        return accs.mean() * 100, accs.std() * 100, toc - tic

    def run(self):
        self.best_epoch = 0
        self.best_val_acc = 0
        self.create_summary_csv = True

        val_acc, val_std, val_t = self.evaluate(self.args.n_val_episodes, self.data.get_val_episode)
        print('{:3d}, meta-val: {:.2f}% ± {:.2f} | {:.1f}s'.format(0, val_acc, val_std, val_t))
        # return
        for epoch in range(self.args.num_epochs):
            train_acc, train_std, train_t = self.train(self.args.n_train_episodes, self.data.get_train_episode)
            val_acc, val_std, val_t = self.evaluate(self.args.n_val_episodes, self.data.get_val_episode)
            print('{:3d}, meta-train: {:.2f}% ± {:.2f} | {:.1f}s, meta-val: {:.2f}% ± {:.2f} | {:.1f}s'.format(
                epoch + 1, train_acc, train_std, train_t, val_acc, val_std, val_t
            ))

            if self.best_val_acc <= val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                torch.save(self.state_dict(), 'best_model.pt')

            epoch_summary = {
                'epoch': epoch + 1, 
                'train_acc': '{:.2f}'.format(train_acc), 'train_std': '{:.2f}'.format(train_std), 
                'val_acc': '{:.2f}'.format(val_acc), 'val_std': '{:.2f}'.format(val_std), 
            }
            if self.create_summary_csv:
                save_statistics(list(epoch_summary.keys()), filename='train_summary_statistics.csv', create=True)
                self.create_summary_csv = False
            save_statistics(list(epoch_summary.values()), filename='train_summary_statistics.csv')

        torch.save(self.state_dict(), 'last_model.pt')
        last_acc, last_std, test_t = self.evaluate(self.args.n_test_episodes, self.data.get_test_episode)
        print('last epoch: {}, meta-val: {:.2f}% meta-test: {:.2f}% ± {:.2f} | {:.1f}s'.format(epoch + 1, val_acc, last_acc, last_std, test_t))

        self.load_state_dict(torch.load('best_model.pt'))
        test_acc, test_std, test_t = self.evaluate(self.args.n_test_episodes, self.data.get_test_episode)
        print('best epoch: {}, meta-val: {:.2f}% meta-test: {:.2f}% ± {:.2f} | {:.1f}s'.format(self.best_epoch, self.best_val_acc, test_acc, test_std, test_t))
        
        test_summary = {
            'params': '{:.2f}'.format(sum(p.numel() for p in self.model.parameters())), 
            'epoch': self.best_epoch, 
            'best_val': '{:.2f}'.format(self.best_val_acc), 
            'test_acc': '{:.2f}'.format(test_acc), 'test_std': '{:.2f}'.format(test_std), 
            'last_acc': '{:.2f}'.format(last_acc), 'last_std': '{:.2f}'.format(last_std), 
        }
        save_statistics(list(test_summary.keys()), filename='test_summary_statistics.csv', create=True)
        save_statistics(list(test_summary.values()), filename='test_summary_statistics.csv')


@hydra.main(config_path='conf/config.yaml', strict=True)
@time_monitor
def main(args):
    device = set_device(args.device)
    set_seed(args.seed)
    exp = Experiment(args, device)
    exp.run()


if __name__ == '__main__':
    main()
