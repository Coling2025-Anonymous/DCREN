# encoding:utf-8
import os
import nni
import math
import time
import json
import torch
import argparse
import torch.nn.functional as F
from joblib._multiprocessing_helpers import mp
from torch.distributions import Categorical
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from nni.utils import merge_parameter

from model import BertEncoder, Classifier, Continuous_learner, RLAgent
from data_process import FewRelProcessor, tacredProcessor
from optimizer import GlobalAdam
from utils import collate_fn, save_checkpoint, get_prototypes, memory_select, set_random_seed, compute_cos_sim, \
    get_augmentative_data, extract_data
import warnings
warnings.filterwarnings("ignore")

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"


def do_train( args, tokenizer, processor, i_exp, global_agent, optimizer, save=False):
    # torch.manual_seed(888 + index)

    # best_memory_acc = None
    best_memory = None
    best_memory_acc = None
    prev_task_acc = []
    prev_memory_acc = []
    prev_max_memory_acc = []
    #state 没经过bert编码
    for j in range(args.epoch_num_RL):

        task_acc = []
        memory_acc = []
        memory = []  # 重放缓存区
        memory_len = []
        relations = []
        prev_classifier, prev_learner = None, None
        taskdatas = processor.get()
        rel2id = processor.get_rel2id()  # {"rel": id}

        # prev_state = torch.zeros((9,768))
        testset = []

        print("Experiment Num{} & Reinforcement Learning epoch{}".format(i_exp, j))

        # global_agent.load_state_dict(global_agent.state_dict())



        h_0 = torch.zeros((1, 16), dtype=torch.float).to(args.device)
        c_0 = torch.zeros((1, 16), dtype=torch.float).to(args.device)

        log_policies = []
        values = []
        rewards = []
        entropies = []



        for i in range(args.task_num):
            # print some info


            task = taskdatas[i]
            traindata, _, testdata = task['train'], task['val'], task['test']
            train_len = task['train_len']
            testset += testdata
            new_relations = task['relation']
            relations += new_relations
            args.seen_rel_num = len(relations)

            print(f"{yellow_print}Training task {i}, relation set {task['relation']}.{default_print}")

            current_encoder = BertEncoder(args, tokenizer, encode_style=args.encode_style)
            current_classifier = Classifier(args, args.hidden_dim, len(new_relations), prev_classifier).to(args.device)



            # get state
            if i == 0:
                # prev_s_vec = get_prototypes(args, current_encoder, traindata, train_len)
                pass
            else:
                # state matrix
                # s = get_prototypes(args, prev_learner, memory[-40:], memory_len[-4:])
                # s =torch.mean(s,dim=0).unsqueeze(0)
                # prev_state[i-1] = s
                # state = prev_state

                prev_proto = get_prototypes(args, prev_learner, memory, memory_len)
                crr_proto = get_prototypes(args, prev_learner, traindata, train_len)

                sim = F.cosine_similarity(crr_proto.unsqueeze(1), prev_proto.unsqueeze(0), dim=-1, eps=1e-08)
                state =  torch.mean(sim, dim=1).unsqueeze(0)

                # state = torch.mean(s, dim=0).unsqueeze(0)

                # state = s_vec*0.6 + state*0.4
                # prev_s_vec = s_vec


                # x = torch.mean(extract_data(memory[-40:]), dim=0).unsqueeze(0).to(args.device)
                # x = F.normalize(x, dim=1)
                # state = torch.cat((state , x), dim=-1)




                a1_logits, a2_logits, a3_logits, value, h_0, c_0  = global_agent(state.to(args.device), h_0, c_0)

                policy1 = F.softmax(a1_logits, dim=1)
                policy2 = F.softmax(a2_logits, dim=1)
                policy3 = F.softmax(a3_logits, dim=1)

                m = Categorical(policy1)#采样
                action1 = m.sample().item()
                m = Categorical(policy2)#采样
                action2 = m.sample().item()
                m = Categorical(policy3)#采样
                action3 = m.sample().item()
                # action1 = torch.argmax(policy1).item()
                # action2 = torch.argmax(policy2).item()
                # action3 = torch.argmax(policy3).item()

                log_policy1 = F.log_softmax(a1_logits, dim=1)
                log_policy2 = F.log_softmax(a2_logits, dim=1)
                log_policy3 = F.log_softmax(a3_logits, dim=1)

                log_policy = (log_policy1[0, action1] + log_policy2[0, action2] + log_policy3[0, action3])/3

                entropy1 = -(policy1 * log_policy1).sum(1, keepdim=True)#计算当前熵值
                entropy2 = -(policy2 * log_policy2).sum(1, keepdim=True)  # 计算当前熵值
                entropy3 = -(policy3 * log_policy3).sum(1, keepdim=True)  # 计算当前熵值

                entropy = (entropy1 + entropy2 + entropy3)/3

                actions = list()
                actions.append(action1)
                actions.append(action2)
                actions.append(action3)




            # train and val on task data

            # memory += new_memory
            # memory_len += new_memory_len

            if prev_learner is not None:
                cl_learner = Continuous_learner(args, args.dataset_name,i_exp,i, prev_learner).to(args.device)
                learner = cl_learner(actions)

            else:
                learner = current_encoder



            if args.dataset_name == "FewRel":
                learner = train_val_task(args, learner, current_classifier, traindata, testdata, rel2id, train_len)
            else:
                aug_traindata = get_augmentative_data(args, traindata, train_len)
                learner = train_val_task(args, learner, current_classifier, aug_traindata, testdata, rel2id, train_len)

            # memory select
            print(f'{blue_print}Selecting memory for task {i}...{default_print}')
            new_memory, new_memory_len = memory_select(args, learner, traindata, train_len)
            memory += new_memory
            memory_len += new_memory_len

            # tmp = get_prototypes(args, learner, memory[-40:], memory_len[-1:])
            # prev_s_vec = tmp*0.6 + prev_s_vec*0.4


            # evaluate on task testdata
            acc = evaluate(args, learner, current_classifier, testdata, rel2id)
            print(f'{blue_print}Accuracy of task {i} is {acc}.{default_print}')
            task_acc.append(acc)

            if j==0:
                reward1 = -0.01
                prev_task_acc.append(acc)

            else:
                reward1 = (acc - prev_task_acc[i])*10
                prev_task_acc[i] = acc

            # train and val on memory data
            if prev_learner is not None:
                print(f'{blue_print}Training on memory...{default_print}')

                current_model = (learner, current_classifier)
                prev_model = (prev_learner, prev_classifier)
                aug_memory = get_augmentative_data(args, memory, memory_len)
                learner = train_val_memory(args, current_model, prev_model, memory, aug_memory, testset, rel2id)
            else:
                print(f"{blue_print}Initial task, won't train on memory.{default_print}")



            # test
            print(f'{blue_print}Evaluating...{default_print}')
            if prev_learner is not None:
                acc = evaluate(args, learner, current_classifier, testset, rel2id)
            else:
                acc = evaluate(args, learner, current_classifier, testset, rel2id)
            print(f'{green_print}Evaluate finished, final accuracy over task 0-{i} is {acc}.{default_print}')
            memory_acc.append(acc)

            if j==0:
                reward2 = -0.01
                prev_memory_acc.append(acc)
                prev_max_memory_acc.append(acc)
            else:
                reward2 = (acc - prev_memory_acc[i])*10 + (acc - prev_max_memory_acc[i])*10
                prev_memory_acc[i] = acc
                if prev_max_memory_acc[i]<acc:
                     prev_max_memory_acc[i] = acc



            if save:
                # save checkpoint
                print(f'{blue_print}Saving checkpoint of task {i}...{default_print}')
                save_checkpoint(args, learner, i_exp, i, "learner")
                save_checkpoint(args, current_classifier, i_exp, i, "classifier")

            prev_learner = learner
            prev_classifier = current_classifier

            nni.report_intermediate_result(acc)

            if i>0:
                values.append(value)
                log_policies.append(log_policy)
                rewards.append(reward2)
                entropies.append(entropy)

        R = torch.zeros((1, 1), dtype=torch.float).to(args.device)
        gae = torch.zeros((1, 1), dtype=torch.float).to(args.device)  # 额外的处理，为了减小variance
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        # if j==1:
        #     print(1)
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * args.g * args.tau
            gae = gae + reward + args.g * next_value.detach() - value.detach()  # Generalized Advantage Estimator 带权重的折扣项
            next_value = value
            actor_loss = actor_loss + log_policy #* gae
            R = R * args.g + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy




        total_loss = -actor_loss + critic_loss - args.b * entropy_loss
        # print("RL_{}/Loss:{}".format(j, total_loss))
        print(f'{yellow_print}RL_{j}_loss: {total_loss}')

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
        #     if global_param.grad is not None:
        #         break
        #     global_param._grad = local_param.grad



        print(f'{green_print}Result of Exemrient {i_exp} & RL {j}:')
        print(f'task acc: {task_acc}')
        print(f'memory acc: {memory_acc}')
        print(f'Average: {sum(memory_acc)/len(memory_acc)}{default_print}')
        if best_memory_acc == None:
            best_memory_acc = (sum(memory_acc) / len(memory_acc))
            best_memory = memory_acc
            f_task_acc = task_acc
            f_memory_acc = memory_acc
        elif best_memory_acc < (sum(memory_acc) / len(memory_acc)):
            best_memory_acc = (sum(memory_acc) / len(memory_acc))
            best_memory = memory_acc
            f_task_acc = task_acc
            f_memory_acc = memory_acc

        print(f'{blue_print}Result of Exemrient {i_exp} & RL {j}:')
        # print(f'task acc: {task_acc}')
        print(f'best memory acc: {best_memory}')
        print(f'best Average: {best_memory_acc}{default_print}')






        if save:
            if not os.path.exists("./checkpoint/agent"):
                os.makedirs("./checkpoint/agent")
            torch.save(global_agent.state_dict(),
                       os.path.join("./checkpoint/agent", "{}_agent.pkl".format(j)))

        # global_agent.load_state_dict(global_agent.state_dict())
        # torch.save(global_agent.state_dict(),
        #            "{}/agent".format("./agent_checkpoint"))

    return f_task_acc, f_memory_acc


def train_val_task(args, encoder, classifier, traindata, valdata, rel2id, train_len):
    dataloader = DataLoader(traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)

    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': classifier.parameters(), 'lr': args.classifier_lr}
        ], eps=args.adam_epsilon)
    # todo add different learning rate for each layer

    best_acc = 0.0
    for epoch in range(args.epoch_num_task):
        encoder.train()
        classifier.train()
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            hidden, _ = encoder(**inputs)

            inputs = {
                'hidden': hidden,
                'labels': batch[4].to(args.device)
            }
            loss, _ = classifier(**inputs)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    acc = evaluate(args, encoder, classifier, valdata, rel2id)
    best_acc = max(acc, best_acc)
    print(f'Evaluate on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')

    return encoder


def train_val_memory(args, model, prev_model, traindata, aug_traindata, testdata, rel2id):
    enc, cls = model
    # prev_enc, prev_cls = prev_model
    dataloader = DataLoader(aug_traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)

    optimizer = AdamW([
        {'params': enc.parameters(), 'lr': args.encoder_lr},
        {'params': cls.parameters(), 'lr': args.classifier_lr}
        ], eps=args.adam_epsilon)

    # prev_enc.eval()
    # prev_cls.eval()
    best_acc = 0.0
    for epoch in range(args.epoch_num_memory):
        enc.train()
        cls.train()

        for step, batch in enumerate(tqdm(dataloader)):
            enc_inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            hidden, feature = enc(**enc_inputs)
            # with torch.no_grad():
            #     prev_hidden, prev_feature = prev_enc(**enc_inputs)

            labels = batch[4].to(args.device)
            # cont_loss = contrastive_loss(args, feature, labels, prototypes, proto_features, prev_feature)
            # cont_loss.backward(retain_graph=True)

            celoss, logits = cls(hidden, labels)

            # epoch_loss+=celoss
            # rep_loss = replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels)
            celoss.backward()

            optimizer.step()
            optimizer.zero_grad()
        # epoch_loss = epoch_loss/step
        # t_loss += epoch_loss

        if (epoch+1) % 10 == 0:
            acc = evaluate(args, enc, cls, testdata, rel2id)
            best_acc = max(best_acc, acc)
            print(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
            nni.report_intermediate_result(acc)

            # prototypes_replay, proto_features_replay = get_prototypes(args, enc, traindata, memory_len)
            # prototypes, proto_features = (1-args.beta)*task_prototypes + args.beta*prototypes_replay, (1-args.beta)*task_proto_features + args.beta*proto_features_replay
            # prototypes = F.layer_norm(prototypes, [args.hidden_dim])
            # proto_features = F.normalize(proto_features, p=2, dim=1)

    return enc





def evaluate(args, model, classifier, valdata, rel2id):
    model.eval()
    dataloader = DataLoader(valdata, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    pred_labels, golden_labels = [], []

    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }

        with torch.no_grad():
            hidden, feature = model(**inputs)
            logits = classifier(hidden)[0]
            final_prob = F.softmax(logits, dim=1)
            # if proto_features is not None:
            #     logits = torch.mm(feature, proto_features.T) / args.cl_temp
            #     prob_ncm = F.softmax(logits, dim=1)
            #     final_prob = args.alpha*prob_cls + (1-args.alpha)*prob_ncm
            # else:
            #     final_prob = prob_cls

        # get pred_labels
        pred_labels += torch.argmax(final_prob, dim=1).cpu().tolist()
        golden_labels += batch[4].tolist()

    pred_labels = torch.tensor(pred_labels, dtype=torch.long)
    golden_labels = torch.tensor(golden_labels, dtype=torch.long)
    acc = float(torch.sum(pred_labels==golden_labels).item()) / float(len(golden_labels))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoint", type=str)
    parser.add_argument("--dataset_name", default="FewRel", type=str)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--cuda_device", default=3, type=int)
    parser.add_argument('--g', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--b', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--plm_name", default="./plm", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--epoch_num_task", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--epoch_num_RL", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--epoch_num_memory", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--hidden_dim", default=768 , type=int, help="Output dimension of encoder.")
    parser.add_argument("--feature_dim", default=64, type=int, help="Output dimension of projection head.")
    parser.add_argument("--encoder_lr", default=1e-5, type=float, help="The initial learning rate of encoder for AdamW.")
    parser.add_argument("--classifier_lr", default=1e-3, type=float, help="The initial learning rate of classifier for AdamW.")
    parser.add_argument("--agent_lr", default=1e-3, type=float, help="The initial learning rate of agent for AdamW.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--alpha", default=0.6, type=float, help="Bagging Hyperparameter.")
    parser.add_argument("--beta", default=0.2, type=float, help="Prototype weight.")
    parser.add_argument("--cl_temp", default=0.1, type=float, help="Temperature for contrastive learning.")
    parser.add_argument("--cl_lambda", default=0.8, type=float, help="Hyperparameter for contrastive learning.")
    parser.add_argument("--margin", default=0.15, type=float, help="Hyperparameter for margin loss.")
    parser.add_argument("--kd_temp", default=0.5, type=float, help="Temperature for knowledge distillation.")
    parser.add_argument("--kd_lambda1", default=0.7, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--kd_lambda2", default=0.5, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--gamma", default=2.0, type=float, help="Hyperparameter of focal loss.")
    parser.add_argument("--encode_style", default="emarker", type=str, help="Encode style of encoder.")
    parser.add_argument("--save_interval", default=10, type=int)

    parser.add_argument("--experiment_num", default=5, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--set_task_order", default=True, type=bool)
    parser.add_argument("--read_from_task_order", default=True, type=bool)
    parser.add_argument("--task_num", default=10, type=int)
    parser.add_argument("--memory_size", default=10, type=int, help="Memory size for each relation.")
    parser.add_argument("--early_stop_patient", default=10, type=int)

    args = parser.parse_args()

    if args.cuda:
        device = "cuda:"+str(args.cuda_device)
    else:
        device = "cpu"

    args.device = device
    args.collate_fn = collate_fn

    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)

    tokenizer = BertTokenizer.from_pretrained(args.plm_name, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

    s = time.time()
    task_results, memory_results = [], []

    for i in range(args.experiment_num):

        set_random_seed(args)
        if args.dataset_name == "FewRel":
            processor = FewRelProcessor(args, tokenizer)
        else:
            processor = tacredProcessor(args, tokenizer)
        if args.set_task_order:
            processor.set_task_order("task_order.json", i)
        if args.read_from_task_order:
            processor.set_read_from_order(i)


        global_agent = RLAgent()
        global_agent.to(args.device)
        # global_agent.share_memory()

        optimizer = AdamW(global_agent.parameters(), lr=args.agent_lr, eps=args.adam_epsilon)



        task_acc, memory_acc = do_train( args, tokenizer, processor, i, global_agent, optimizer,True)
        print(f'{green_print}Result of experiment {i}:')
        print(f'task acc: {task_acc}')
        print(f'memory acc: {memory_acc}')
        print(f'Average: {sum(memory_acc) / len(memory_acc)}{default_print}')
        task_results.append(task_acc)
        memory_results.append(memory_acc)
        # mp.set_start_method('spawn')
        # processes = []
        # for index in range(2):
        #     if index == 0:
        #         process = mp.Process(target=do_train, args=(index, args, tokenizer, processor, i, global_agent, optimizer, True))
        #     else:
        #         process = mp.Process(target=do_train, args=(index, args, tokenizer, processor, i, global_agent, optimizer))
        #     process.start()
        #     processes.append(process)
        # # process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
        # # process.start()
        # # processes.append(process)
        # for process in processes:
        #     process.join()
        # mp.set_start_method('spawn')

        # print(f'{green_print}Result of experiment {i}:')
        # print(f'task acc: {task_acc}')
        # print(f'memory acc: {memory_acc}')
        # print(f'Average: {sum(memory_acc)/len(memory_acc)}{default_print}')


        # task_results.append(task_acc)
        # memory_results.append(memory_acc)
            # torch.cuda.empty_cache()
    e = time.time()

    task_results = torch.tensor(task_results, dtype=torch.float32)
    memory_results = torch.tensor(memory_results, dtype=torch.float32)
    print(f'All task result: {task_results.tolist()}')
    print(f'All memory result: {memory_results.tolist()}')

    task_results = torch.mean(task_results, dim=0).tolist()
    memory_results = torch.mean(memory_results, dim=0)
    final_average = torch.mean(memory_results).item()
    print(f'Final task result: {task_results}')
    print(f'Final memory result: {memory_results.tolist()}')
    print(f'Final average: {final_average}')
    print(f'Time cost: {e-s}s.')

    nni.report_final_result(final_average)