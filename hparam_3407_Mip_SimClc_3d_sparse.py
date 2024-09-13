class hparams:

    fold = 'seed_3047'#0 # defalut = 0
    train_or_test = 'train'
    # output_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/ckpt'%fold
    # ckpt_dir = '/data/shanwenqi/0Vessel_seg/TransUnet_Mip/Mip_3d_sparse_prompt_right/ckpt/checkpoint_0288.pt'
    ckpt_dir = '/data/shanwenqi/0Vessel_seg/TransUnet_Mip/checkpoint_0236.pt'
    # output_dir = '/data/shanwenqi/0Vessel_seg/TransUnet_Mip/Mip_3d_sparse_prompt_right/ckpt'
    output_dir = '/data/shanwenqi/0Vessel_seg/TransUnet_Mip/Mip_3d_sparse_prompt_right_multi_add/ckpt'
    # output_dir = '/data/shanwenqi/Vessel_seg/TransUnet_Mip/Mip_3d_sparse_multi_add/ckpt'
    # output_dir = 'logs/batch2'
    aug = False
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 300
    epochs_per_checkpoint = 1
    batch_size = 1
    ckpt = None
    init_lr = 0.0003 # 0.01
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    # crop_or_pad_size = 64,64,64 # if 2D: 256,256,1
    patch_size = 192,192,64 # if 2D: 128,128,1 
    # patch_size = 192,192,1 # if 2D: 128,128,1 
    # patch_size = 512,512,1 # if 2D: 128,128,1 
    # patch_size = 512,512,1 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0
    # patch_overlap = 4,4,0 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'
    data_description = 'image_align_direction_hdbet_N4_corrected'
    # model_nsetting = 'image_align_direction_hdbet'
    # source_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/fold_%s/no_skull/image'%fold
    
    # label_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/fold_%s/no_skull/label'%fold
    # source_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/fold_%s/no_skull/image'%fold
    # label_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/fold_%s/no_skull/label'%fold
    
    # source_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_img_64slice'
    # label_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_64slice'
    
    source_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/noskull_N4_image'
    label_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/noskull_N4_label'
    # mip_pred_to_3d_sparse_mask_train_dir = '/memory/shanwenqi/datasets/nnUNet_raw_data/nnUNet_raw_data/Task815_MIP/Tr_pred_3d_sparse'
    mip_pred_to_3d_sparse_mask_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_pred_3d_sparse_right_IXI'
    
    source_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image'
    label_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_label'
    mip_pred_to_3d_sparse_mask_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_pred_3d_sparse_right_IXI'

 
    grad_clip = 0.3
    r1_lambda = 0.5

    # output_int_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/pred/int_pred'%fold
    # output_float_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/pred/float_pred'%fold 
    # output_int_dir = '/data/shanwenqi/Vessel_seg/TransUnet_Mip/Mip_3d_sparse_multi_add/pred/int_pred'
    output_int_dir = '/data/shanwenqi/0Vessel_seg/TransUnet_Mip/Mip_3d_sparse_prompt_right_multi_add/pred/int_pred'
    output_float_dir = '/data/shanwenqi/0Vessel_seg/TransUnet_Mip/Mip_3d_sparse_prompt_right_multi_add/pred/float_pred'

    init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
    cost_weight = [2.0, 3.0, 5.0]
            # matcher = HungarianMatcher3D(
            #     cost_class=cost_weight[0], # 2.0
            #     cost_mask=cost_weight[1],
            #     cost_dice=cost_weight[2],
            # )
# class hparams:
#     fold = 2 # defalut = 0
#     train_or_test = 'train'
#     output_dir = '/data/shanwenqi/Vessel_seg/ISA/fold_%s'%fold
#     # output_dir = 'logs/batch2'
#     aug = False
#     latest_checkpoint_file = 'checkpoint_latest.pt'
#     total_epochs = 100
#     epochs_per_checkpoint = 5
#     batch_size = 4
#     ckpt = None
#     init_lr = 0.01
#     scheduer_step_size = 20
#     scheduer_gamma = 0.8
#     debug = False
#     mode = '3d' # '2d or '3d'
#     in_class = 1
#     out_class = 1

#     crop_or_pad_size = 64,64,64 # if 2D: 256,256,1
#     patch_size = 64,64,64 # if 2D: 128,128,1 

#     # for test
#     patch_overlap = 4,4,4 # if 2D: 4,4,0

#     fold_arch = '*.nii.gz'

#     save_arch = '.nii.gz'

#     source_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/fold_%s/image'%fold
#     label_train_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/fold_%s/label'%fold
#     source_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/fold_%s/image'%fold
#     label_test_dir = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/fold_%s/label'%fold
    
#     # source_train_dir = '/data/cc/TOF/GAN/train/source'
#     # label_train_dir = '/data/cc/TOF/GAN/train/label'
#     # source_test_dir = '/data/cc/TOF/GAN/test/source'
#     # label_test_dir = '/data/cc/TOF/GAN/test/label'
	
	
# 	# source_train_dir = '/data/cc/Ying-TOF/train/source'
#     # label_train_dir = '/data/cc/Ying-TOF/train/label1'
#     # source_test_dir = '/data/cc/Ying-TOF/test/source'
#     # label_test_dir = '/data/cc/Ying-TOF/test/label1'

#     grad_clip = 0.3
#     r1_lambda = 0.5

#     output_int_dir = '/data/shanwenqi/Vessel_seg/ISA/fold_%s/pred/int_pred'%fold
#     output_float_dir = '/data/shanwenqi/Vessel_seg/ISA/fold_%s/pred/float_pred'%fold

#     init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]