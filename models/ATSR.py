# ATSR model
import torch
import torch.nn as nn
from models.modeling_bart import BartModel, BartPretrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple
from opt_einsum import contract
from models.modeling_roberta_ import RobertaModel_, RobertaPreTrainedModel
from models.modeling_roberta_ import *
from models.process_long import process_long_input_decode, process_long_input
import torch.nn.functional as F
import json
import csv
import random

class ArgumentExpert(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()  

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)  

class ArgumentAwareMoE(nn.Module):
    def __init__(self, hidden_size, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        self.experts = nn.ModuleList([
            ArgumentExpert(hidden_size) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(hidden_size, num_experts)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        gate_score = self.gating(x)  
        topk_val, topk_idx = torch.topk(gate_score, self.k, dim=-1) 
        expert_mask = torch.zeros(x.size(0), self.num_experts, device=x.device)
        expert_mask.scatter_(1, topk_idx, 1.0)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_expert_mask = expert_mask * gate_score 
        output = torch.sum(expert_outputs * weighted_expert_mask.unsqueeze(-1), dim=1)

        return output, gate_score

class GatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)


class ATSR(RobertaPreTrainedModel):
    def __init__(self, config, decode_layer_start=17,tokenizer=None):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer

        self.template_dict, self.argument_dict = self._read_roles(self.config.role_path)

        decode_layer_start = config.encoder_layers

        self.roberta = RobertaModel_(config, decode_layer_start=decode_layer_start)

        self.w_prompt_start = nn.Parameter(torch.rand(config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.rand(config.hidden_size, ))

        self.num_templates = 3
        self.w_template_select = nn.Parameter(torch.randn(self.num_templates, config.hidden_size))

        self.decode_layer_start = decode_layer_start

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')
        self.contextual_merger = nn.Linear(2 * config.hidden_size, config.hidden_size)
    
        # moe
        self.residual_c = config.residual_c
        self.arg_res = config.arg_res

        self.role_name_mapping = None
        if self.config.dataset == 'MLEE':
            with open(self.config.role_path) as f:
                self.role_name_mapping = json.load(f)
        
        if self.config.use_arg_moe:
            self.argument_slot_moe = ArgumentAwareMoE(
                hidden_size=config.hidden_size,
                num_experts=self.config.moe_num_experts,
                k=self.config.moe_top_k
            )
        else:
            self.argument_slot_moe = None

    def load_balancing_loss_func_per_sample(self, gate_logits_list_per_sample, num_experts, top_k):
        batch_lb_losses = []
        
        for sample_gates in gate_logits_list_per_sample:
            if len(sample_gates) == 0:
                continue
            
            sample_gate_logits = torch.cat(sample_gates, dim=0)  # [num_slots_in_sample, num_experts]
            
            routing_weights = F.softmax(sample_gate_logits, dim=-1)
            topk_val, topk_idx = torch.topk(routing_weights, top_k, dim=-1)
            
            expert_mask = torch.zeros_like(routing_weights)
            expert_mask.scatter_(1, topk_idx, 1.0)
            
            tokens_per_expert = expert_mask.mean(dim=0)  # [num_experts]
            router_prob_per_expert = routing_weights.mean(dim=0)  # [num_experts]
            
            lb_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts
            batch_lb_losses.append(lb_loss)

        if len(batch_lb_losses) > 0:
            return torch.stack(batch_lb_losses).mean()
        else:
            return torch.tensor(0.0, device=sample_gate_logits.device)

    def reset(self):
        self.w_prompt_start = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.roberta._init_weights(self.contextual_merger.weight)

    def context_pooling(self, value_matrix, trigger_att, hidden_rep):  

        att = (value_matrix* trigger_att)
        att = att / (att.sum(1, keepdim=True) + 1e-5) 
        rs = contract("ld,rl->rd", hidden_rep, att)

        return rs
        
    def select_template(self, trigger_repr):
        scores = torch.matmul(trigger_repr, self.w_template_select.t()) 
        template_weights = torch.softmax(scores, dim=0)
        selected_idx = torch.argmax(template_weights).item()
        
        return template_weights, selected_idx
    
    def _read_roles(self, role_path):
        template_dict = {}
        role_dict = {}

        if 'MLEE' in role_path:
            with open(role_path) as f:
                role_name_mapping = json.load(f)
                for event_type, mapping in role_name_mapping.items():
                    roles = list(mapping.keys())
                    role_dict[event_type] = roles

            return None, role_dict

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template

                event_type, arg = event_type_arg.split('_')
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)

        return template_dict, role_dict

    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        all_ids=None,
        all_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        arg_list=None,
        event_triggers=None,
        enc_attention_mask=None,
        template_options=None,
        event_types=None,
        argument_dict=None,
        offset_prompt=None,
    ):
        """
        Args:
            multi args post calculation
        """

        context_outputs_ = self.roberta(
        input_ids=all_ids,
        attention_mask=all_mask_ids,
        output_hidden_states=True,
        fully_encode=True,
        output_attentions=True,
        return_dict=True,
    )  
        
        enc_outputs = context_outputs_.hidden_states
        decoder_context = enc_outputs[self.decode_layer_start]

        if self.config.context_representation == 'decoder':
            context_outputs = enc_outputs[-1]
        else:
            context_outputs = decoder_context

        encoder_attentions = context_outputs_.attentions[self.decode_layer_start].mean(1)

        selected_dec_prompt_ids = []
        selected_dec_prompt_mask_ids = []
        selected_arg_joint_prompts = []  
        all_selected_idx = {}

        for batch_idx, (event_trigger, event_template_options, sample_event_types) in enumerate(zip(event_triggers, template_options, event_types)):           
    
            batch_dec_prompt_ids = []
            batch_dec_prompt_mask_ids = []
            batch_arg_slots = []  
            one_sample_idx = []
            
            event_type_to_template = {}  
            added_templates = set()  
            current_prompt_offset = 0  
            
            for event_idx, (trigger_pos, event_type) in enumerate(zip(event_trigger, sample_event_types)):
                if event_type in event_type_to_template:
                    selected_idx, selected_template_text, template_prompt_offset = event_type_to_template[event_type]
                    one_sample_idx.append(selected_idx)
                else:
                    trigger_repr = torch.mean(
                        decoder_context[batch_idx][trigger_pos[0]:trigger_pos[1]], 
                        dim=0
                    )
                    template_weights, selected_idx = self.select_template(trigger_repr)
                    selected_template_text = event_template_options[event_idx][selected_idx]
                    event_type_to_template[event_type] = (selected_idx, selected_template_text, current_prompt_offset)

                    one_sample_idx.append(selected_idx)
    
                selected_idx, selected_template_text, template_prompt_offset = event_type_to_template[event_type]
                
                template_key = (selected_template_text, selected_idx)
                if template_key not in added_templates:
                    added_templates.add(template_key)
                    
                    selected_tokens = self.tokenizer(selected_template_text, add_special_tokens=True)
                    selected_ids = selected_tokens["input_ids"]
                    selected_masks = selected_tokens["attention_mask"]
                    
                    batch_dec_prompt_ids.extend(selected_ids)
                    batch_dec_prompt_mask_ids.extend(selected_masks)
                    
                    current_prompt_offset += len(selected_ids)

                template_relative_offset = template_prompt_offset 
                template_tokens = self.tokenizer(selected_template_text, add_special_tokens=True)
                arg_list = self.argument_dict[event_type.replace(':', '.')]
                arg_2_prompt_slots = dict()
                
                for arg in arg_list:
                    prompt_slots = {
                        "tok_s": list(), "tok_e": list(),
                        "tok_s_off": list(), "tok_e_off": list(),
                    }

                    if self.config.dataset == 'MLEE':
                        arg_ = self.role_name_mapping[event_type][arg]
                    else :
                        arg_ = arg

                    import re
                  
                    for matching_result in re.finditer(r'\b' + re.escape(arg_) + r'\b',
                                                    selected_template_text.split('.')[0]):
                        char_idx_s, char_idx_e = matching_result.span()
                        char_idx_e -= 1
                        tok_prompt_s = template_tokens.char_to_token(char_idx_s)
                        tok_prompt_e = template_tokens.char_to_token(char_idx_e) + 1
                        
                      
                        actual_tok_s = tok_prompt_s + template_relative_offset
                        actual_tok_e = tok_prompt_e + template_relative_offset
                        
                        actual_tok_s_off = actual_tok_s + offset_prompt[batch_idx]
                        actual_tok_e_off = actual_tok_e + offset_prompt[batch_idx]
                        
                        prompt_slots["tok_s"].append(actual_tok_s)
                        prompt_slots["tok_e"].append(actual_tok_e)
                        prompt_slots["tok_s_off"].append(actual_tok_s_off)
                        prompt_slots["tok_e_off"].append(actual_tok_e_off)
                    
                    arg_2_prompt_slots[arg] = prompt_slots
                
              
                batch_arg_slots.append(arg_2_prompt_slots)
            
         
            all_selected_idx[batch_idx] = one_sample_idx

         
            while len(batch_dec_prompt_ids) < self.config.max_prompt_seq_length:
                batch_dec_prompt_ids.append(self.tokenizer.pad_token_id)
                batch_dec_prompt_mask_ids.append(0)
                
            selected_dec_prompt_ids.append(batch_dec_prompt_ids)
            selected_dec_prompt_mask_ids.append(batch_dec_prompt_mask_ids)
            selected_arg_joint_prompts.append(batch_arg_slots)

        selected_dec_prompt_ids = torch.tensor(selected_dec_prompt_ids, device=all_ids.device)
        selected_dec_prompt_mask_ids = torch.tensor(selected_dec_prompt_mask_ids, device=all_mask_ids.device)
        
        decoder_prompt_outputs = self.roberta(
            input_ids=selected_dec_prompt_ids,
            attention_mask=selected_dec_prompt_mask_ids,
            encoder_hidden_states=decoder_context,
            encoder_attention_mask=all_mask_ids,
            cross_attention=True,
        ).last_hidden_state 

        
        gate_logits_list_per_sample = [[] for _ in range(len(context_outputs))]  

        logit_lists = list()
        total_loss = 0.
        if len(event_triggers) == 0:
            print(len(event_triggers))
        for i, (context_output, decoder_prompt_output, encoder_attention, selected_arg_joint_prompt,arg_joint_prompt, old_tok_to_new_tok_index, event_trigger) in \
            enumerate(zip(context_outputs, decoder_prompt_outputs, encoder_attentions,selected_arg_joint_prompts,arg_joint_prompts, old_tok_to_new_tok_indexs, event_triggers)):
            
            batch_loss = list()
            cnt = 0
            list_output = list()
            for ii in range(len(event_trigger)):
                
                event_trigger_pos = event_trigger[ii]
                event_trigger_attention = torch.mean(encoder_attention[event_trigger_pos[0]:event_trigger_pos[1]], dim=0).unsqueeze(0)
            
                output = dict()
                for arg_role in selected_arg_joint_prompt[ii].keys():
                 
                    """
                    "arg_role": {"tok_s": , "tok_e": }
                    """
                    prompt_slots = selected_arg_joint_prompt[ii][arg_role]

                    # encoder :
                    prompt_slots_enc = arg_joint_prompt[ii][arg_role]
                    count=0
                    #
                    start_logits_list = list()
                    end_logits_list = list()
                    for (p_start,p_end, p_start_off, p_end_off) in zip(prompt_slots['tok_s'], prompt_slots['tok_e'], prompt_slots['tok_s_off'], prompt_slots['tok_e_off']):
                        enc_text_prompt_slots = prompt_slots_enc['tok_s_off'][count]
                        enc_text_prompt_slote = prompt_slots_enc['tok_e_off'][count]
                        if self.config.dataset == 'wikievent':
                            prompt_query_sub = decoder_prompt_output[p_start:p_end]
                        
                        if prompt_query_sub.shape[0] == 0:
                            prompt_query_sub = context_output[0].unsqueeze(0)
                      
                        if enc_text_prompt_slots >= self.config.max_enc_seq_length  or enc_text_prompt_slote >= self.config.max_enc_seq_length:
                            prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                        else:
                         
                            prompt_query_sub_attention = encoder_attention[enc_text_prompt_slots:enc_text_prompt_slote]
                        
                            if prompt_query_sub_attention.shape[0] == 0:
                                prompt_query_sub_attention = encoder_attention[0]
                          
                            prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                            prompt_query_sub_attention = torch.mean(prompt_query_sub_attention, dim=0).unsqueeze(0)
                            
                            if self.argument_slot_moe is not None:
                                prompt_query_sub_enhanced, gate_score1 = self.argument_slot_moe(prompt_query_sub)
                                gate_logits_list_per_sample[i].append(gate_score1)
                                prompt_query_sub = prompt_query_sub + self.arg_res * prompt_query_sub_enhanced
                     
                           
                            context_rs = self.context_pooling(prompt_query_sub_attention, event_trigger_attention, decoder_context[i])
                            
                            prompt_query_sub = torch.tanh(self.contextual_merger(torch.cat((prompt_query_sub, context_rs), dim=-1)))
                        
                  
                        count+=1 
                        start_query = (prompt_query_sub*self.w_prompt_start).unsqueeze(-1) # [1, H, 1]
                        end_query = (prompt_query_sub*self.w_prompt_end).unsqueeze(-1)     # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)      

                    output[arg_role] = [start_logits_list, end_logits_list]

                    if self.training:
                       
                        target = target_info[i][ii][arg_role] 
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        target_spans = [[s,e] for (s,e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans)<len(predicted_spans):
                       
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0,0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                        batch_loss.append((start_loss + end_loss)/2)
                list_output.append(output)
            logit_lists.append(list_output)
            if self.training: # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss))/cnt
            
        if self.training:
            total_loss_ = total_loss/len(context_outputs)
            if self.argument_slot_moe is not None:
                lb_loss = self.load_balancing_loss_func_per_sample(
                    gate_logits_list_per_sample=gate_logits_list_per_sample,
                    num_experts=self.config.moe_num_experts,
                    top_k=self.config.moe_top_k
                )
                total_loss_ = total_loss_ + self.config.lambd * lb_loss

            return total_loss_, logit_lists
        else:
            return [], logit_lists

