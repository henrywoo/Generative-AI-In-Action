from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
device = "cuda" # the device to load the model onto
model_path = "Qwen/Qwen1.5-0.5B"  #根据自身具备的算力条件进行自适应更改，如切换为1.5B、7B甚至更大，本处我们照顾在消费级显卡上进行全参微调，选择最小的0.5B版本

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [151645, 151643]
model.generation_config.pad_token_id = 151643
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.transformers_version = "4.37.0"
model.generation_config.repetition_penalty = 1.05

from dataclasses import dataclass
@dataclass
class modelConfig:
    max_length:int = 550 #根据自身具备的算力条件进行自适应更改
    batch_size:int = 2
    gradient_accumulation_steps:int = 1
    alpha:float = 0.2
    log_iter:int = 500
    max_lr:float = 1e-5
    min_lr:float = 5e-7
    warmup_steps:int = 2000

import datasets
binarized_data = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized')

def tokenize_and_format(data): # data List[Dict[str, str]]
        input_ids = tokenizer.apply_chat_template(
    data, # Union[List[Dict[str, str]]
    tokenize = True,
    add_generation_prompt = False,
    #padding = True,
    truncation = True,
    max_length = modelConfig.max_length,
)

        return input_ids #tag_ids

## 生成偏好数据的tokenid
chosen_input_ids_list = []
#tag_ids_list = []
i = 0
while True:
    data = binarized_data['train_sft'][i]['chosen']
    data.insert(0, {"content": "You are a helpful assistant", "role": "system"})
    input_ids = tokenize_and_format(data)
    chosen_input_ids_list.append(input_ids)
    #tag_ids_list.append(tag_ids)
    i += 1
    if i%10000 == 0 or i== len(binarized_data['train_sft']):
        print(f"偏好数据已处理{i}条数据")
    if i ==25000:#len(binarized_data['train_sft']): ##先只取前25000条数据用于训练测试对比
        break
print('-'*70)
#############################################################################
## 生成不偏好数据的tokenid
rejected_input_ids_list = []
#tag_ids_list = []
i = 0
while True:
    data = binarized_data['train_sft'][i]['rejected']
    data.insert(0, {"content": "You are a helpful assistant", "role": "system"})
    input_ids = tokenize_and_format(data)
    rejected_input_ids_list.append(input_ids)
    #tag_ids_list.append(tag_ids)
    i += 1
    if i%10000 == 0 or i== len(binarized_data['train_sft']):
        print(f"非偏好数据已处理{i}条数据")
    if i ==25000:# len(binarized_data['train_sft']): ##先只取前25000条数据用于训练测试对比
        break

assert len(chosen_input_ids_list) == len(rejected_input_ids_list)  ##确保数据条数一致

batch_size = modelConfig.batch_size
gradient_accumulation_steps = modelConfig.gradient_accumulation_steps
log_iter = modelConfig.log_iter
alpha = modelConfig.alpha
max_lr = modelConfig.max_lr
min_lr = modelConfig.min_lr
warmup_steps = modelConfig.warmup_steps
total_steps = len(chosen_input_ids_list)//batch_size
optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=max_lr)
trainable_parameters_num = sum(p.numel() for p in filter(lambda p:p.requires_grad, model.parameters()))  ##全参微调

##配置logging
import time
#import logging
with open("orpo_result/orpo_log.txt", "w+") as my_file:
  my_file.write(f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, batch_size:{batch_size}, trainable_parameters_num:{trainable_parameters_num}, warmup_steps:{warmup_steps}, max_lr:{max_lr}, min_lr:{min_lr}\n')
#定义一个日志记录函数
def log_call(iters, iters_average_loss, iters_average_logprobability_for_Positive, iters_average_logprobability_for_Negetive, iters_average_Log_Odds_Ratio, iters_average_Log_Odds):
  with open("orpo_result/orpo_log.txt", "w+") as my_file:
    my_file.write(f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, iters:{iters+1}, iters_average_Loss:{iters_average_loss:.4f}, iters_average_logprobability_for_Positive:{iters_average_logprobability_for_Positive:.4f}, iters_average_logprobability_for_Negetive:{iters_average_logprobability_for_Negetive:.4f}, iters_average_Log_Odds_Ratio:{iters_average_Log_Odds_Ratio:.4f}, iters_average_Log_Odds:{iters_average_Log_Odds:.4f}\n')


def linear_warmup(current_step, warmup_steps, max_lr):
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps
    else:
        return max_lr

def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
    if current_step < warmup_steps:
        return linear_warmup(current_step, warmup_steps, max_lr)
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return (max_lr - min_lr) * decay + min_lr

def return_answer_mask(input_ids):
  assistant_answer_mask = torch.zeros_like(input_ids)
  for i in range(input_ids.shape[0]):
        ## user部分的结尾\n: \n是<|im_end|>的下一个元素，所以有+1 【这个地方需要根据不同模型的不同聊天模版自定义更改】，关于聊天模版可阅读这篇文章：https://huggingface.co/blog/chat-templates
        i_user_end_list = [i+1 for i in torch.where(input_ids[i]==tokenizer.encode('<|im_end|>')[0])[0].tolist()[1::2]]
        ## assistant部分的结尾\n：\n是<|im_end|>的下一个元素，所以有+1 【这个地方需要根据不同模型的不同聊天模版自定义更改】
        i_assistant_end_list = [i+1 for i in torch.where(input_ids[i]==tokenizer.encode('<|im_end|>')[0])[0].tolist()[2::2]]

        if len(i_user_end_list)==len(i_assistant_end_list):
            for user_end, assistant_end in zip(i_user_end_list, i_assistant_end_list):
                assistant_answer_mask[i][user_end+3:assistant_end+1]=1 #+3的操作，【这个地方需要根据不同模型的不同聊天模版自定义更改】
        elif len(i_user_end_list)==len(i_assistant_end_list)+1==1:  ##单轮问答,且回答部分未结尾就被截断了
            assistant_answer_mask[i][i_user_end_list[0]+3:]=1  ##会把右补的padding token也标记为1，所以后面还需要再结合padding mask以过滤padding
        elif len(i_user_end_list)==len(i_assistant_end_list)+1:   ##兼顾多轮问答，且最后一轮对话的回答部分还未结尾就被截断了，此处我们只看完整回答部分的损失
            for user_end, assistant_end in zip(i_user_end_list[:-1], i_assistant_end_list):
                assistant_answer_mask[i][user_end+3:assistant_end+1]=1
        else:
            continue  ##跳出当前循环，继续下一次循环
  return assistant_answer_mask

model.train()
train_loss_list = []
pos_prob_list = []
neg_prob_list = []
Log_Odds_Ratio_list = []
Log_Odds_list = []
model.zero_grad() ##clear gradients at the start of training
ignore_iters_count = 0
for iters in range(len(chosen_input_ids_list)//batch_size):
    ## 获取批次数据
    chosen_batch_inputids = chosen_input_ids_list[iters*batch_size:(iters+1)*batch_size]
    rejected_batch_inputids = rejected_input_ids_list[iters*batch_size:(iters+1)*batch_size]

    ## 对该批次数据进行padding,以并行计算，首先计算该批次的最大token长度
    chosen_max_dim = max([len(i) for i in chosen_batch_inputids])
    rejected_max_dim = max([len(i) for i in rejected_batch_inputids])
    ### 偏好数据padding填充
    chosen_batch_inputids_padding_list = []
    for i in range(batch_size):
        chosen_batch_inputids_padding_list.append(torch.nn.functional.pad(torch.tensor(chosen_batch_inputids[i]), (0, chosen_max_dim - len(chosen_batch_inputids[i])), mode='constant', value=model.generation_config.eos_token_id[-1]).tolist()) #右补
    chosen_batch_inputids_tensor = torch.tensor(chosen_batch_inputids_padding_list)
    ### 非偏好数据padding填充
    rejected_batch_inputids_padding_list = []
    for i in range(batch_size):
        rejected_batch_inputids_padding_list.append(torch.nn.functional.pad(torch.tensor(rejected_batch_inputids[i]), (0, rejected_max_dim - len(rejected_batch_inputids[i])), mode='constant', value=model.generation_config.eos_token_id[-1]).tolist()) #右补
    rejected_batch_inputids_tensor = torch.tensor(rejected_batch_inputids_padding_list)

    ## 构建训练数据：x->y ,下一个单词预测
    chosen_x = chosen_batch_inputids_tensor[:, :-1].to(device)
    chosen_y = chosen_batch_inputids_tensor[:, 1:].to(device)
    rejected_x = rejected_batch_inputids_tensor[:, :-1].to(device)
    rejected_y = rejected_batch_inputids_tensor[:, 1:].to(device)

    ## 构建掩码判别矩阵（paddding mask & answer_mask, padding mask用于执行对padding的token不计算损失，answer_mask用于执行仅针对回答部分才计算损失），总之，就是确认哪些tokens的logit需要"忽视"掉
    ### 【padding mask】
    chosen_padding_mask = torch.where(chosen_y==model.generation_config.eos_token_id[-1], 0, 1)
    rejected_padding_mask = torch.where(rejected_y==model.generation_config.eos_token_id[-1], 0, 1)
    ### 【answer_mask】
    chosen_assistant_answer_mask = return_answer_mask(chosen_x)
    rejected_assistant_answer_mask = return_answer_mask(rejected_x)
    ### 【paddingmask & answermask】方便使用掩码判别矩阵对logit和y进行过滤->:我们只关注【回答】部分的损失，不关注问题部分的损失
    chosen_assistant_answer_mask = (chosen_assistant_answer_mask&chosen_padding_mask)
    rejected_assistant_answer_mask = (rejected_assistant_answer_mask&rejected_padding_mask)

    if chosen_assistant_answer_mask.sum(dim=-1).min().item()==0 or rejected_assistant_answer_mask.sum(dim=-1).min().item()==0:##如果该批次里有的问答数据在数据截取时，回答部分存在未有数据的情况，那么该批次数据不再训练
      #print(f'不处理第{iters+1}批次数据')
      ignore_iters_count+=1
      continue  #跳出当前循环

    ## 执行偏好数据的模型推理，计算logits
    chosen_logits = model(chosen_x).logits
    torch.cuda.empty_cache() #清除非必要的显存占用，但会导致速度变慢

    ## 执行非偏好数据的模型推理，计算logits
    rejected_logits = model(rejected_x).logits
    torch.cuda.empty_cache() #清除非必要的显存占用，但会导致速度变慢

    """ORPO论文链接 https://arxiv.org/abs/2403.07691"""
    ## Compute Chosen_Answer_Loss，计算偏好数据的回答部分的损失, pos_loss的shape_size:[batch_size]
    pos_loss = torch.mul((torch.gather(torch.log(torch.softmax(chosen_logits, dim=-1)), dim=-1, index=chosen_y.unsqueeze(2))*(-1)).squeeze(2), chosen_assistant_answer_mask).sum(dim=-1) / chosen_assistant_answer_mask.sum(dim=-1)
    ## Calculate Log Probability, pos_prob、neg_prob的shape_size:[batch_size]
    pos_prob = torch.mul((torch.gather(torch.log(torch.softmax(chosen_logits, dim=-1)), dim=-1, index=chosen_y.unsqueeze(2))).squeeze(2), chosen_assistant_answer_mask).sum(dim=-1) / chosen_assistant_answer_mask.sum(dim=-1)
    neg_prob = torch.mul((torch.gather(torch.log(torch.softmax(rejected_logits, dim=-1)), dim=-1, index=rejected_y.unsqueeze(2))).squeeze(2), rejected_assistant_answer_mask).sum(dim=-1) / rejected_assistant_answer_mask.sum(dim=-1)
    ## Calculate log odds
    log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob)))
    sig_ratio = torch.nn.functional.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    ## Calculate the Final Loss, 参考论文的方式，只是新增了梯度积累的操作
    loss = torch.nanmean(pos_loss - alpha * ratio)/(gradient_accumulation_steps)

    loss.backward()

    # Compute the learning rate for the current step
    lr = cosine_decay(iters, warmup_steps, total_steps, max_lr, min_lr)

    # Update the learning rate for the AdamW optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (iters+1)%gradient_accumulation_steps==0 or (iters+1)==(len(chosen_input_ids_list)//batch_size):
        optimizer.step() #update weights after gradients accumulation
        ##at last, clear gradients
        optimizer.zero_grad() #clear gradients after updating, in this case equal to model.zero_grad()

    train_loss_list.append(loss.item()*gradient_accumulation_steps)
    pos_prob_list.append(torch.nanmean(pos_prob).item())
    neg_prob_list.append(torch.nanmean(neg_prob).item())
    Log_Odds_Ratio_list.append(torch.nanmean(ratio).item())
    Log_Odds_list.append(torch.nanmean(log_odds).item())
    if (iters+1)%log_iter==0 or (iters+1)==(len(chosen_input_ids_list)//batch_size):
      print(f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, iters:{iters+1}, last_{log_iter}_iters_average_train_Loss:{np.nanmean(train_loss_list[-log_iter:]):.4f}, last_{log_iter}_iters_average_logprobability_for_Positive:{np.nanmean(pos_prob_list[-log_iter:]):.4f}, last_{log_iter}_iters_average_logprobability_for_Negetive:{np.nanmean(neg_prob_list[-log_iter:]):.4f}, last_{log_iter}_iters_average_Log_Odds_Ratio:{np.nanmean(Log_Odds_Ratio_list[-log_iter:]):.4f}, last_{log_iter}_iters_average_Log_Odds:{np.nanmean(Log_Odds_list[-log_iter:]):.4f}') ##避免空值影响
      log_call(iters, np.nanmean(train_loss_list[-log_iter:]), np.nanmean(pos_prob_list[-log_iter:]), np.nanmean(neg_prob_list[-log_iter:]), np.nanmean(Log_Odds_Ratio_list[-log_iter:]), np.nanmean(Log_Odds_list[-log_iter:]))
print("Totally Completed!")
print(f'共计忽略{ignore_iters_count}个批次数据')

torch.save(model.state_dict(),'orpo_result/'+model_path.split('/')[1]+'_ORPO.pth')