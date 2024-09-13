class hparams:

    fold = 'seed_3047'#0 # defalut = 0
    train_or_test = 'train'
    ckpt_dir = './checkpoint_0236.pt'
    output_dir = './SPNet/ckpt'
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
    patch_size = 128,128,64 # if 2D: 128,128,1 
    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'
    save_arch = '.nii.gz'
    data_description = 'ADAM'
    
    source_train_dir = './ADAM/train/img_mra'
    label_train_dir = './ADAM/train/lbl_mra'
    mip_pred_to_3d_sparse_mask_train_dir = './ADAM/train/mip_pred_3d_sparse_mask'
    
    source_test_dir = './ADAM/test/img_mra'
    label_test_dir = './ADAM/test/lbl_mra'
    mip_pred_to_3d_sparse_mask_test_dir = './ADAM/test/mip_pred_3d_sparse_mask'
 
    grad_clip = 0.3
    r1_lambda = 0.5

    output_int_dir = './SPNet/pred'

    init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
    cost_weight = [2.0, 5.0, 5.0]


#     output_int_dir = '/data/shanwenqi/Vessel_seg/ISA/fold_%s/pred/int_pred'%fold
#     output_float_dir = '/data/shanwenqi/Vessel_seg/ISA/fold_%s/pred/float_pred'%fold

#     init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]
