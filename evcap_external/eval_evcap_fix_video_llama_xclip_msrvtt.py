import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from search import beam_search
import random
import numpy as np

from models.evcap_fix_video_llama_xclip_msrvtt import EVCap 

# from dataset.msrvtt_dataset import MSRVTTDataset
# from dataset.msrvtt_dataloader.msrvtt_dataloader import make_data_loader
from dataset.msrvtt.vl_dataloader import make_data_loader, build_dataset
from dataset.msrvtt.data_utils.miscellaneous import str_to_bool

from torchvision import transforms
from collections import OrderedDict
from datasets import load_dataset
from einops import rearrange

import time

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validation_msrvtt(
    args,
    test_yaml, 
    model,
    tokenizer, 
) -> None:

    device = args.device
    
    test_dataloader, _ = make_data_loader(args, test_yaml, None, is_distributed=False, is_train=False)    
    if not args.load_video_feature: 
        train_dataset = build_dataset(args, 'datasets/MSRVTT-v2/train_16frames.yaml', None, is_train=True)
    else:
        train_dataset = None
    # with open(args.ext_path, 'r') as file:
    #     retri_cap = json.load(file)
    predicts = []
    temp = None
    for idx, item in tqdm(enumerate(test_dataloader)): 
        image_id = item['image_id']
        captions = item['text_input']
        if len(captions[0]) == 1:
            captions = [cap[0] for cap in captions]
        if len(image_id) == 1:
            image_id = image_id[0]
        image= item['image'].to(device)
        if idx == 0:
            temp = image_id
        elif image_id == temp:
            continue
        else:
            temp = image_id
        print('\n')
        print('Image ID: ', image_id)
        print('GT: ', captions)
        # print('retrieval captions: ', retri_cap[image_id])

        with torch.cuda.amp.autocast(enabled=True):
            image = rearrange(image, 'b f c h w -> (b f) c h w')
            # qform_all_proj, atts_qform_all_proj = model.encode_img(image, [image_id], train_dataset, bs=1)
            qform_all_proj, atts_qform_all_proj = model.encode_img(image, [image_id])
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list) 
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model) # List[str]
            sentence = sentence[0]
            print('Pred: ', sentence)

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f'Prediction time for video {image_id}: {elapsed_time:.2f} seconds')
        # break

        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    #out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json') #default
    # outfile = args.ckpt_name[:-3].replace('train', 'eval')
    outfile = 'eval' + args.ckpt_name[5:-3]
    out_json_path = os.path.join(args.out_path, f'{outfile}_generated_captions.json')
    # if args.retriever:
    #     out_json_path = out_json_path.replace('_generated', '_retriever_generated')
    print(f'generated_captions saved as: {out_json_path}')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)

@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    # loading model
    
    # by lijian
    # model_type = "vicuna-13b-v1.3" # 内存不够，跑不了
    model_type = "vicuna-7b-v1.3"
    outpath = args.out_path
    outpath = outpath.replace('eval', 'train')
    model = EVCap(
            ext_path = args.ext_path,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            # freeze_qformer_txt=args.freeze_qformer_txt, # train txtQFormer
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn = args.topn,
            llama_model=model_type,
            prompt_path=args.prompt_path,
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=128,
            end_sym='\n',
            low_resource=False,
            device_8bit=0,
            bs=args.bs,
            frames=args.max_num_frames,
            retriever=args.retriever,
            num_video_query_token=args.num_video_query_token, # video-llama
            freeze_video_qformer=args.freeze_video_qformer, # video-llama
            freeze_qformer_txt=args.freeze_qformer_txt, # video-llama
            # load_video_feature=args.load_video_feature
        )
    if args.ckpt_name is not None:
        # ckpt = os.path.join(outpath, args.ckpt_name) # 'outputs/train_evcap_msrvtt/000.pt'
        ckpt = args.ckpt_name
        print('load:', ckpt)
        state_dict = torch.load(ckpt, map_location=device)['model']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    test_yaml = args.test_yaml
    tokenizer = model.llama_tokenizer
    validation_msrvtt(args, test_yaml, model, tokenizer)


if __name__ == '__main__':
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--out_path', default = 'outputs/eval_evcap_fix_video_llama_msrvtt')
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 9)
    parser.add_argument('--name_of_datasets', default = 'msrvtt')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    
    # by lijian
    parser.add_argument('--ckpt_name', default = None)
    #msrvtt
    parser.add_argument('--test_yaml', default = 'datasets/MSRVTT-v2/llx_test_32frames.yaml')
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument("--num_workers", default=1, type=int, help="Workers in dataloader.")
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument("--on_memory", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Option to load labels/caption to memory before training.")
    parser.add_argument("--img_feature_dim", default=512, type=int,
                            help="Update model config if given.The Image Feature Dimension.")
    parser.add_argument("--debug_speed", type=str_to_bool, default=False)
    parser.add_argument('--retriever', type=str_to_bool, nargs='?', const=True, default=True, help = '是否需要检索模块和Customized Q-Former')
    # video-llama
    parser.add_argument('--num_video_query_token', type = int, default = 32)
    parser.add_argument("--freeze_video_qformer", type=str_to_bool, default=True)
    # XCLIP
    parser.add_argument("--ext_path", default='ext_data/msrvtt_train_caption.json', type=str)
    # prompt
    parser.add_argument("--prompt_path", default='prompts/prompt_evcap_msrvtt.txt', type=str)
    
    parser.add_argument("--freeze_qformer_txt", type=str_to_bool, default=True)
    parser.add_argument('--load_video_feature', type=str_to_bool, default=True)
        
    args = parser.parse_args()
    set_seed(args.random_seed)
    print('args: {}\n'.format(vars(args)))
    
    main(args)
    