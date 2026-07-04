import torch
import os
import numpy as np
import gc

from ..protein import constants
from ._base import register_transform

from esm.pretrained import load_model_and_alphabet_local

def ESM_encoder(dir, proname, structure, model, alphabet, batch_converter, device):
    aa_number=structure['aa']
    resindex_to_ressymb = {v: k for k, v in constants.ressymb_to_resindex.items()}
    # print(aa_number)
    # print(resindex_to_ressymb)
    # assert False,"debug"
    aa_symbols = [resindex_to_ressymb[num.item()] for num in aa_number]
    seq=''.join(aa_symbols)
    seq = seq.replace('*', 'X')
    data = [("protein1", seq)]
    os.makedirs(dir, exist_ok=True)
    llm_path = os.path.join(dir, proname + '.npz')
    if os.path.exists(llm_path):
        try:
            feature = np.load(llm_path)
            llm = feature['llm']
            # contact = feature['contact']
            assert llm.shape[0] == len(seq),"ESM shape wrong!"
            # assert contact.shape[0] == len(seq) and contact.shape[1] == len(seq),"ESM contact shape wrong!"
            return torch.FloatTensor(llm)
        except Exception as e:
            print(e)
            print("Reload ESM feature for %s" % proname)

    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            
        token_repr = results["representations"][33].detach().clone().cpu()  # (B, L+2, D)
        # contacts = results["contacts"].detach().clone().cpu()
        del results,batch_tokens,batch_labels,batch_strs
        gc.collect()

        # contact_arr = contacts[0, :, :].numpy()  # (L, L)
        arr = token_repr[0, 1:-1, :].numpy()
        del token_repr
        gc.collect()
        # assert contact_arr.shape[0]==len(seq) ,f'pre esm contact shape wrong {contact_arr.shape[0]}!={len(seq)}'
        assert arr.shape[0]==len(seq) ,f'pre esm shape wrong {arr.shape[0]}!={len(seq)}'

        np.savez(llm_path, llm=arr)
    except Exception as e:
        print(f"{proname} error->{e}")
        return None,None
    return torch.FloatTensor(arr)

@register_transform('merge_chains')
class MergeChains(object):

    def __init__(self):
        super().__init__()
        ##import  esm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device=torch.device("cpu") #lzh
        self.model, self.alphabet = load_model_and_alphabet_local(
            "/home/data/user/lizonghui/feature/esm2_t33_650M_UR50D.pt"
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model = self.model.to(self.device)
        ##

    def assign_chain_number_(self, data_list):
        chains = set()
        for data in data_list:
            chains.update(data['chain_id'])
        chains = {c: i for i, c in enumerate(chains)}

        for data in data_list:
            data['chain_nb'] = torch.LongTensor([
                chains[c] for c in data['chain_id']
            ])

    def _data_attr(self, data, name):
        if name in ('generate_flag', 'anchor_flag') and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        else:
            return data[name]

    def __call__(self, structure):
        data_list = []
        if structure['heavy'] is not None:
            structure['heavy']['fragment_type'] = torch.full_like(
                structure['heavy']['aa'],
                fill_value = constants.Fragment.Heavy,
            )
            data_list.append(structure['heavy'])

        if structure['light'] is not None:
            structure['light']['fragment_type'] = torch.full_like(
                structure['light']['aa'],
                fill_value = constants.Fragment.Light,
            )
            data_list.append(structure['light'])

        if structure['antigen'] is not None:
            structure['antigen']['fragment_type'] = torch.full_like(
                structure['antigen']['aa'],
                fill_value = constants.Fragment.Antigen,
            )
            structure['antigen']['cdr_flag'] = torch.zeros_like(
                structure['antigen']['aa'],
            )
            data_list.append(structure['antigen'])

        self.assign_chain_number_(data_list)
        ##
        self.dirh="/xsdata/lzhlzh/antibody-diffusion-properties-main/esm/h"
        self.dirl="/xsdata/lzhlzh/antibody-diffusion-properties-main/esm/l"
        self.dira="/xsdata/lzhlzh/antibody-diffusion-properties-main/esm/a"
        self.allname=structure['id']
        # print(self.allname)
        self.pname=self.allname.split('_')[0]     
        # if structure['heavy'] is  None:       #???????????????????
        #     print('HHHH')
        # if structure['light'] is  None:
        #     print('LLLL')
        # if structure['antigen'] is  None:
        #     print('AAAA')
        if structure['heavy'] is not None:
            self.hchain=self.allname.split('_')[1]
            structure['heavy']['llm']=ESM_encoder(self.dirh,self.pname+'_'+self.hchain,structure['heavy'],self.model,self.alphabet,self.batch_converter,self.device)   #lzh
        if structure['light'] is not None:
            self.lchain=self.allname.split('_')[2]
            structure['light']['llm']=ESM_encoder(self.dirl,self.pname+'_'+self.lchain,structure['light'],self.model,self.alphabet,self.batch_converter,self.device)   #lzh
        if structure['antigen'] is not None:
            self.achain=self.allname.split('_')[3]
            structure['antigen']['llm']=ESM_encoder(self.dira,self.pname+'_'+self.achain,structure['antigen'],self.model,self.alphabet,self.batch_converter,self.device)   #lzh
        ##

        list_props = {
            'chain_id': [],
            'icode': [],
        }
        tensor_props = {
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generate_flag': [],
            'cdr_flag': [],
            'anchor_flag': [],
            'fragment_type': [],
            'llm': [],
        }

        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(self._data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(data, k))

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {
            **list_props,
            **tensor_props,
        }
        return data_out

