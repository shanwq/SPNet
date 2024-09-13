class hparams:

    fold = 1 # defalut = 0
    train_or_test = 'train'
    # output_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/ckpt'%fold
    ckpt_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/ICASSP/TransUnet/ckpt/checkpoint_0125.pt'
    output_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/ICASSP/TransUnet/ckpt'
    # output_dir = 'logs/batch2'
    aug = False
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 200
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

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'
    data_description = 'image_align_direction_hdbet_N4_corrected'
    # model_nsetting = 'image_align_direction_hdbet'
    
    source_train_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/train/img_N4'
    label_train_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/train/lbl_N4'
    source_test_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/test/img_N4'
    label_test_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/test/lbl_N4'
	
    grad_clip = 0.3
    r1_lambda = 0.5

    # output_int_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/pred/int_pred'%fold
    # output_float_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/fold_%s/pred/float_pred'%fold 
    output_int_dir = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/ICASSP/TransUnet/pred'
    output_float_dir = '/data/shanwenqi/Vessel_seg/ISA_TransUnet3d/seed3047/pred_new/float_pred'

    init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
    cost_weight = [2.0, 3.0, 5.0]
            # matcher = HungarianMatcher3D(
            #     cost_class=cost_weight[0], # 2.0
            #     cost_mask=cost_weight[1],
            #     cost_dice=cost_weight[2],
            # )
