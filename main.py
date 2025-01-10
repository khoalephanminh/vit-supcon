import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# prompt_model = generate_model_resnet(model_depth = 50, include_top=True)
# prompt_model.load_state_dict(torch.load('/workspace/train_deep_prompt/model/contrastive_prompt/promptGen50_ckp.pt'))
# prompt_model.projection_head = nn.Conv3d(2048, 768, kernel_size=1, stride=1)

# model = PromptedVisionTransformer(
#     image_size=160,
#     image_patch_size=16,
#     frames = 120,
#     frame_patch_size = 12,
#     num_layers=12,
#     num_heads=12,
#     hidden_dim=768,
#     mlp_dim=3072,
#     dropout=0.1,
#     emb_dropout=0.1,
#     channels = 1,
#     num_classes = 5,
#     freeze_vit = True,
#     pool = 'cls',
#     pretrain_path = '/workspace/train_deep_prompt/pretrained/jx_vit_base_p16_224-80ecf9dd.pth',
#     num_prompts = 8,
#     prompt_dropout = 0.1,
# )# .to(device)
# model.to(device)



# count_freeze = 0
# count_tuning = 0
# tuning_params = []
# freeze_params = []
# for name, param in model.named_parameters():
#     # if 'prompt' in name or 'mlp_head' in name:
#     if param.requires_grad == True:
#         count_tuning += 1
#         tuning_params.append(name)
#     else:
#         count_freeze += 1
#         #freeze_params.append(name)
# print(f'There are {count_tuning} trainable params.')
# print(f'including: {tuning_params}')
# print(f'There are {count_freeze} freeze params')
# # print(f'including: {freeze_params}')


# # criterion_cls = FocalLoss(
# #     gamma=2,
# #     alpha=0.25)
# # criterion_cont = SupervisedContrastiveLoss()

# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-4,)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0 = 15,
#     T_mult = 2,
#     eta_min=1e-6
# )



# checkpoint_path = '/workspace/train_deep_prompt/model/contrastive_prompt/imagenet/best_checkpoint.pth'
# checkpoint = torch.load(checkpoint_path)

# model.load_state_dict(checkpoint['model_state_dict'])
# if 'prompt_model_state_dict' in checkpoint:
#     prompt_generator.load_state_dict(checkpoint['prompt_model_state_dict'])

# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# if checkpoint['scheduler_state_dict']:
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])