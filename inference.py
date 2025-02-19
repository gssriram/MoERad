import torch
import os
import json 
from tqdm import tqdm 
import tiktoken
from models.MoE import MoERad, GPTConfig, cap_dataset, beam_search


import argparse
parser = argparse.ArgumentParser(description='get_entities')
parser.add_argument('--input_json_file', type=str,default='data/iu_xray/ReXRank_IUXray_test.json')
parser.add_argument('--save_json_file', type=str,default='results/iu_xray/MoERad.json')
parser.add_argument('--img_root_dir', type=str,default='/teamspace/studios/relieved-sapphire-uabn/R2Gen/data/iu_xray/images/')

args = parser.parse_args()

# Setting up preliminaries
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'=> Proceeding with {device}')
enc = tiktoken.get_encoding('gpt2')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')

# Setting up the model
model = MoERad(GPTConfig(), dropout_p=0.0)
model.to(device)

# Loading the checkpoints
chkpt = torch.load('models/MoE_model.pt')
state_dict = chkpt['model']

for name in list(state_dict.keys()):
    state_dict[name[len('_orig_mod.'):]] = state_dict[name]
    
    del state_dict[name]

model.load_state_dict(chkpt['model'], strict=True)
print('=> Model Checkpoint loaded successfully')


# Load the dataset
dataset = cap_dataset()
input_json_file = args.input_json_file
img_root_dir =  args.img_root_dir
save_json_file = args.save_json_file

with open(input_json_file, 'r') as file:
    input_data_dict = json.load(file)

if os.path.exists(save_json_file):
    with open(save_json_file, 'r') as file:
        save_data_dict = json.load(file)
else:
    save_data_dict = {}

for study_id in tqdm(input_data_dict.keys()):
    if study_id in save_data_dict:
        pass 
    else:
        data_dict_idx = input_data_dict[study_id]
        save_data_dict_idx = data_dict_idx
        image_path_list = data_dict_idx['key_image_path']
        img_path = img_root_dir+image_path_list
        img = dataset.preprocess_image(img_path)
        
        img = img.to(device)
        start_tok = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})

        report_tokens = beam_search(
                model=model,
                img_features=img,
                start_token_id=start_tok[0],
                end_token_id=0,
                max_len=256,
                beam_width=3,
                num_beam_groups=3,    # Number of diverse groups
                diversity_penalty=1.0  # Diversity strength
            )

        decoded = enc.decode(report_tokens)
        pred_report = decoded.split('<|endoftext|>')[1]

        save_data_dict_idx['model_prediction'] = pred_report
        save_data_dict[study_id] = save_data_dict_idx


    with open(save_json_file, 'w') as file:
        json.dump(save_data_dict, file, indent=4)

    