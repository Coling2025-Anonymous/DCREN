import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_cos_sim, extract_data
import collections
from torch.nn import CrossEntropyLoss
from transformers import BertModel
import torch.nn.functional as F


class RLAgent(nn.Module):

    def __init__(self):
        super(RLAgent, self).__init__()


        # self.struct = nn.LSTM(input_size=1, hidden_size=3, num_layers=1, bias=False, batch_first=True)


        self.struct1 = nn.LSTMCell(8, 16)
        self.struct2 = nn.LSTMCell(8, 16)
        self.struct3 = nn.LSTMCell(8, 16)
        # self.fc = nn.Linear(512, 256)
        # self.norm = torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
        # self.actions_vec = nn.Parameter(torch.randn(3, 256, 2))  # dim0 : 模型的层数  dim1: hiddensize dim2: 动作数量
        # nn.init.xavier_normal_(self.actions_vec)
        self.softmax= nn.Softmax(dim=-1 )
        self.flatten = nn.Flatten()
        self.critic_linear = nn.Linear(16, 1)
        self.actor_linear1 = nn.Linear(16, 6) # 3 动作数量
        self.actor_linear2 = nn.Linear(16, 6)
        self.actor_linear3 = nn.Linear(16, 6)


        # self.prev_state = None

    def forward(self, state, h0, c0):
        # state = (c_input, p_input)  #怎么计算状态，然后lstm 将 state作为h0，c0  和c_input一块输入模型采样动作
        # state = state.unsqueeze(-1)
        # if state.size(1) != 256:
        #     state = self.fc(state)

        # state = F.normalize(state , dim=1)

        h1, c1 = self.struct1(state,(h0, c0))
        vec1 = h1.reshape(1,-1)
        a1 = self.actor_linear1(vec1)

        h2, c2 = self.struct2(state, (h1, c1))
        vec2 = h2.reshape(1, -1)
        a2 = self.actor_linear1(vec2)

        h3, c3 = self.struct3(state, (h2, c2))
        vec3 = h3.reshape(1, -1)
        a3 = self.actor_linear1(vec3)

        #
        # a3 = self.actor_linear3(vec)
        # v = (vec1+vec2+vec3)/3

        v1= self.critic_linear(vec1)
        v2 = self.critic_linear(vec2)
        v3 = self.critic_linear(vec3)

        v = (v1+v2+v3)/3
        # action_probs = self.softmax(action_logits)
        # actions = np.argmax(action_probs.data.type(torch.FloatTensor).numpy(),axis=1)
        # self.prev_state = c_n

        return a1, a2, a3, v , h3, c3

class Continuous_learner(nn.Module):
    def __init__(self, args, dataset_name, i_exp, task_num, prev_learner):
        super().__init__()
        self.args = args
        self.learner = prev_learner
        self.Krepo = None
        self.layers = ['top.0','middle.0','bottom.0']
        self.task_num =task_num
        self.dataset_name = dataset_name
        self.i_exp = i_exp

        # self.path= './checkpoint/tacred/Exp0'
        # self.hidden_dim = hidden_dim


    def forward(self, actions=None):
        self.learner = self.recover(self.learner)

        #实际动作
        for action,layer in zip(actions,self.layers):
            if action == 0:  #action:load
                # self.learner = self.load(self.learner, layer)
                print("exec load action")   #动态更新模型
            elif action == 1: #action: fuse
                self.learner = self.fuse( self.learner, layer)
                print("exec fuse action")
            elif action == 2:
                self.learner = self.new_fc(self.learner, layer)
                print("exec new_fc action")
            elif action == 3:
                self.learner = self.reset(self.learner, layer)
                print("exec reset action")
            elif action == 4:
                self.learner = self.remove(self.learner, layer)
                print("exec remove action")
            elif action == 5:
                self.learner = self.protect(self.learner, layer)
                print("exec protect action")


        print("Continuous Learner construction complete!")

        return self.learner

    def load(self, model, layer):
        if self.task_num ==0:
            pass
        else:
            model_dict = model.state_dict()

            learner_path = './checkpoint/{0}/Exp{1}/{2}_learner.pkl'.format(self.dataset_name,self.i_exp,self.task_num-1)
            self.Krepo=torch.load(learner_path)

            state_dict = {k: v for k, v in self.Krepo.items() if k[:k.find(".")] in layer}
            # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        return model

    def fuse(self,  model, layer):
        if self.task_num==0:
            pass
        else:
            state_dict = dict()
            model_dict = model.state_dict()
            for i in range(self.task_num):
                learner_path = './checkpoint/{0}/Exp{1}/{2}_learner.pkl'.format(self.dataset_name, self.i_exp,i)
                self.Krepo = torch.load(learner_path)
                state_dict[i]={k: v for k, v in self.Krepo.items() if k[:k.find(".")] in layer}

            # for i in range(self.num):
            #     state_dict[i]=collections.Counter(state_dict[i])

            for i in range(self.task_num):
                state_dict[i] = collections.Counter(state_dict[i])
                if i==0:
                   p =state_dict[i]
                else:
                   # p = p + state_dict[i]
                   for key, value in state_dict[i].items():
                       if key in p:
                            p[key] += value
                       else:
                            p[key] = value

            p = dict(p)
            for k, v in p.items():
                p[k] = v / self.task_num
                # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            model_dict.update(p)
            model.load_state_dict(model_dict)

        return model

    def new_fc(self , model, layer):
        if self.task_num == 0:
            pass
        else:
            if layer == 'top':
                model.top.add_module(str(len(model.top)+1), nn.Linear(768*2, 768*2).to(self.args.device))
            elif layer == 'middle':
                model.middle.add_module(str(len(model.middle)+1), nn.Linear(768, 768).to(self.args.device))
            elif layer == 'bottom':
                model.bottom.add_module(str(len(model.bottom)+1), nn.Linear(768, 768).to(self.args.device))
        return model


    def reset(self , model, layer):
        if self.task_num == 0:
            pass
        else:
            if layer == 'top':
                # for m in model.top:
                #     if isinstance(m, (nn.Conv2d, nn.Linear)):
                #         nn.init.kaiming_normal_(m.weight, mode='fan_in')

                for m in model.top:
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif layer == 'middle':
                for m in model.top:
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif layer == 'bottom':
                for m in model.top:
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        return model

    def remove(self , model, layer):
        if self.task_num == 0:
            pass
        else:
            if layer == 'top':
                if len(model.top) >1 :
                    net = nn.Sequential(*list(model.top)[:-1])
                    model.top = net
            elif layer == 'middle':
                if len(model.middle) > 1:
                    net = nn.Sequential(*list(model.middle)[:-1])
                    model.middle = net
            elif layer == 'bottom':
                if len(model.bottom) > 3:
                    net = nn.Sequential(*list(model.bottom)[:-1])
                    model.bottom = net
        return model

    def protect(self , model, layer):
        if self.task_num == 0:
            pass
        else:
            if layer == 'top':
                for (name, param) in model.top.named_parameters():
                    param.requires_grad = False

            elif layer == 'middle':
                for (name, param) in model.middle.named_parameters():
                    param.requires_grad = False
            elif layer == 'bottom':
                for (name, param) in model.bottom.named_parameters():
                    param.requires_grad = False
        return model

    def recover(self , model):

        for (name, param) in model.named_parameters():
                param.requires_grad = True

        return model


class BertEncoder(nn.Module):
    def __init__(self, args, tokenizer, encode_style="emarker"):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = BertModel.from_pretrained(args.plm_name).to(args.device)
        self.model.resize_token_embeddings(len(tokenizer))

        # 'cls' using the cls_token as the embedding
        # 'emarker' concatenating the embedding of head and tail entity markers
        if encode_style in ["cls", "emarker"]:
            self.encode_style = encode_style
        else:
            raise Exception("Encode_style must be 'cls' or 'emarker'.")

        if encode_style == "emarker":
            hidden_size = self.model.config.hidden_size
            self.top = nn.Sequential(
                nn.Linear(hidden_size*3, hidden_size*2),
                nn.ReLU(),
                # nn.LayerNorm([hidden_size*2])
            ).to(self.args.device)
            self.middle = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                # nn.LayerNorm([hidden_size])
            ).to(self.args.device)
            self.bottom = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm([hidden_size])
            ).to(self.args.device)


    def forward(self, input_ids, attention_mask, h_index, t_index, labels = None):
        plm_output = self.model(input_ids, attention_mask=attention_mask)
        if self.encode_style == "cls":
            hidden = plm_output['last_hidden_state'].index_select(1, torch.tensor([0]).to(self.args.device)).squeeze()  # [batch_size, hidden_size]
        else:
            h = torch.stack([plm_output['last_hidden_state'][i, h_index[i], :] for i in range(len(h_index))], dim=0) # [batch_size, hidden_size]
            t = torch.stack([plm_output['last_hidden_state'][i, t_index[i], :] for i in range(len(t_index))], dim=0)
            cls = plm_output['pooler_output']
            ht_embeddings = torch.cat([cls, h, t], dim=-1) # [batch_size, hidden_size*2]
            top_out  = self.top(ht_embeddings) # [batch_size, hidden_size]
            middle_out = self.middle(top_out)
            hidden = self.bottom(middle_out) # [batch_size, feature_dim]
            # feature = F.normalize(feature, p=2, dim=1) # [batch_size, feature_dim]

        output = (hidden, hidden)
        
        # if labels is not None:
        #     # compute scloss of current task
        #     dot_div_temp = torch.mm(feature, feature.T) / self.args.cl_temp # [batch_size, batch_size]
        #     dot_div_temp_norm = dot_div_temp - torch.max(dot_div_temp, dim=1, keepdim=True)[0].detach() # [batch_size, batch_size]
        #     exp_dot_temp = torch.exp(dot_div_temp_norm) + 1e-8 # avoid log(0)  [batch_size, batch_size]
        #
        #     mask = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels).to(self.args.device) # [batch_size, batch_size]
        #     cardinalities = torch.sum(mask, dim=1) # [batch_size]
        #
        #     log_prob = -torch.log(exp_dot_temp / torch.sum(exp_dot_temp, dim=1, keepdim=True)) # [batch_size, batch_size]
        #     scloss_per_sample = torch.sum(log_prob*mask, dim=1) / cardinalities # [batch_size]
        #     scloss = torch.mean(scloss_per_sample)
        #
        #     loss = scloss
        #     output = (loss, ) + output

        return output
    
    def get_low_dim_feature(self, hidden):
        feature = self.head(hidden)
        feature = F.normalize(feature, p=2, dim=1)
        return feature


# class Classifier(nn.Module):
#     def __init__(self, args, hidden_dim, label_num, prev_classifier=None):
#         super().__init__()
#         self.args = args
#         self.label_num = label_num
#         self.classifier = nn.Linear(hidden_dim, label_num, bias=False)
#         self.loss_fn = CrossEntropyLoss()
#
#     def forward(self, hidden, labels=None):
#         logits = self.classifier(hidden)
#         output = (logits,)
#
#         if labels is not None:
#             loss = self.loss_fn(logits, labels)
#             output = (loss,) + output
#
#         return output
#
#     def incremental_learning(self, seen_rel_num):
#         weight = self.classifier.weight.data
#         self.classifier = nn.Linear(768, seen_rel_num, bias=False).to(self.args.device)
#         with torch.no_grad():
#             self.classifier.weight.data[:seen_rel_num] = weight[:seen_rel_num]

class Classifier(nn.Module):
    def __init__(self, args, hidden_dim, label_num, prev_classifier=None, state=False):
        super().__init__()
        self.args = args
        self.label_num = label_num
        self.prev_classifier = prev_classifier
        self.state =state
        # self.classifier = nn.Linear(hidden_dim, label_num, bias=False)
        self.classifier_weight = nn.Parameter(torch.empty(hidden_dim,label_num)).to(args.device)
        nn.init.xavier_normal_(self.classifier_weight)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, hidden, labels=None):
        if self.prev_classifier != None and self.state==False:
            self.classifier_weight = torch.cat((self.prev_classifier.classifier_weight, self.classifier_weight), dim=-1)
            self.state = True

        # logits = self.classifier(hidden)
        logits = torch.matmul(hidden,self.classifier_weight)

        output = (logits, )

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output = (loss, ) + output

        return output

    def incremental_learning(self, seen_rel_num):
        weight = self.classifier.weight.data
        self.classifier = nn.Linear(768, seen_rel_num, bias=False).to(self.args.device)
        with torch.no_grad():
            self.classifier.weight.data[:seen_rel_num] = weight[:seen_rel_num]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
