import numpy as np
import SimpleITK as sitk
import os

img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image/'
lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_label/'
mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_img_64slice/'
mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_64slice/'

mip_pred_train_path = '/memory/shanwenqi/datasets/nnUNet_raw_data/nnUNet_raw_data/Task815_MIP/Trpred/'
mip_pred_train_arg = '/memory/shanwenqi/datasets/nnUNet_raw_data/nnUNet_raw_data/Task815_MIP/Tr_arg/'

# os.makedirs(mip_pred_train_arg, exist_ok=True)
# os.makedirs(mip_lbl_16slice_path, exist_ok=True)
def mip_generate():
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        
        # img_npy_path = mip_img_16slice_path + img_name[:-7]
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(lbl)

        mip_img_all = np.zeros((28,1024,1024))
        mip_lbl_all = np.zeros((28,1024,1024))
        print(x.shape)
        for kk in range(0, 28):
            # if kk < 8 :
            #     mip_x = x[0:16,:,:]
            #     mip_y = y[0:16,:,:]
                
            # elif kk > 55 :
            #     mip_x = x[-16:,:,:]
            #     mip_y = y[-16:,:,:]
            # else:
            #     # print(kk-7,kk+9)
            mip_x = x[kk:kk+64,:,:]
            mip_y = y[kk:kk+64,:,:]
                
            print(kk, mip_x.shape)
            mip_x_img = np.max(mip_x, 0)
            mip_y_img = np.max(mip_y, 0)
            print(mip_x_img.shape)
            # quit()
            mip_img = np.expand_dims(mip_x_img, axis=0)
            mip_lbl = np.expand_dims(mip_y_img, axis=0)
            print('mip_img',mip_img.shape)
            print('mip_lbl',mip_lbl.shape)
            # quit()
            mip_img = sitk.GetImageFromArray(mip_img)
            mip_lbl = sitk.GetImageFromArray(mip_lbl)
            # mip_img.CopyInformation(img)
            # mip_lbl.CopyInformation(img)
            
            mip_img.SetDirection(direc)
            mip_lbl.SetDirection(direc)
            mip_img.SetOrigin(origin)
            mip_lbl.SetOrigin(origin)
            mip_img.SetSpacing(spac)
            mip_lbl.SetSpacing(spac)
            
            
            sitk.WriteImage(mip_img, os.path.join(mip_img_16slice_path, '%s'%(img_name[:-7]) + 'img_slice_%s.nii.gz'%kk))
            sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_16slice_path, '%s'%(img_name[:-7]) + 'lbl_slice_%s.nii.gz'%kk))
            
            
            # mip_img_all[kk,:,:] = mip_x_img
            # mip_lbl_all[kk,:,:] = mip_y_img
            # np.save(os.path.join(mip_img_16slice_path, '%s'%(img_name[:-7]) + 'img_slice_%s.npy'%kk), mip_x_img)
            # np.save(os.path.join(mip_lbl_16slice_path, '%s'%(img_name[:-7]) + 'lbl_slice_%s.npy'%kk), mip_y_img)
            
        # quit()
mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_img_allslice/'
mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_allslice/'

# os.makedirs(mip_img_allslice_path, exist_ok=True)
# os.makedirs(mip_lbl_allslice_path, exist_ok=True)
def mip_generate_all():
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        
        # img_npy_path = mip_img_16slice_path + img_name[:-7]
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(lbl)

        mip_img_all = np.zeros((1,1024,1024))
        mip_lbl_all = np.zeros((1,1024,1024))
        print(x.shape)
        for kk in range(0, 1):
            # if kk < 8 :
            #     mip_x = x[0:16,:,:]
            #     mip_y = y[0:16,:,:]
                
            # elif kk > 55 :
            #     mip_x = x[-16:,:,:]
            #     mip_y = y[-16:,:,:]
            # else:
            #     # print(kk-7,kk+9)
            mip_x = x[kk:kk+92,:,:]
            mip_y = y[kk:kk+92,:,:]
                
            print(kk, mip_x.shape)
            mip_x_img = np.max(mip_x, 0)
            mip_y_img = np.max(mip_y, 0)
            print(mip_x_img.shape)
            # quit()
            mip_img = np.expand_dims(mip_x_img, axis=0)
            mip_lbl = np.expand_dims(mip_y_img, axis=0)
            print('mip_img',mip_img.shape)
            print('mip_lbl',mip_lbl.shape)
            # quit()
            mip_img = sitk.GetImageFromArray(mip_img)
            mip_lbl = sitk.GetImageFromArray(mip_lbl)
            # mip_img.CopyInformation(img)
            # mip_lbl.CopyInformation(img)
            
            mip_img.SetDirection(direc)
            mip_lbl.SetDirection(direc)
            mip_img.SetOrigin(origin)
            mip_lbl.SetOrigin(origin)
            mip_img.SetSpacing(spac)
            mip_lbl.SetSpacing(spac)
            
            
            sitk.WriteImage(mip_img, os.path.join(mip_img_allslice_path, '%s'%(img_name[:-7]) + 'img_slice_%s.nii.gz'%kk))
            sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_allslice_path, '%s'%(img_name[:-7]) + 'lbl_slice_%s.nii.gz'%kk))
            
            
            # mip_img_all[kk,:,:] = mip_x_img
            # mip_lbl_all[kk,:,:] = mip_y_img
            # np.save(os.path.join(mip_img_16slice_path, '%s'%(img_name[:-7]) + 'img_slice_%s.npy'%kk), mip_x_img)
            # np.save(os.path.join(mip_lbl_16slice_path, '%s'%(img_name[:-7]) + 'lbl_slice_%s.npy'%kk), mip_y_img)
            
        # quit()
    
def mip_generate_all_args():
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        
        img_arg_npy_path = mip_pred_train_arg + img_name[:-7]
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(lbl)

        print(x.shape)
        # for kk in range(0, 1):
            # mip_x = x[kk:kk+92,:,:]
            # mip_y = y[kk:kk+92,:,:]
            
        img_index = np.argmax(x, 0)
        print(x.shape, img_index.shape)
        print(x.shape, img_index.shape, np.unique(img_index))
        # np.save(img_arg_npy_path + '.npy', img_index)
        
    
# mip_generate_all()
# mip_generate_all_args()

def from_mip_pred_and_args_to_3d_sparse_seg():
    save_path = '/memory/shanwenqi/datasets/nnUNet_raw_data/nnUNet_raw_data/Task815_MIP/Tr_lbl_3d_sparse_right/'
    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        print(img_name)
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        # mip_pred = sitk.ReadImage(mip_pred_train_path + img_name)
        mip_lbl_train_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_labelsTr/labelsTr/'
        
        mip_pred = sitk.ReadImage(mip_lbl_train_path + img_name[:-7] + 'lbl_slice_all.nii.gz')
        mip_arr = sitk.GetArrayFromImage(mip_pred)
        # mip_arr = np.swapaxes(mip_arr, 0,2)
        print('mip_arr', mip_arr.shape, np.unique(mip_arr))
        # quit()
        print('mip_arr.shape', mip_arr.shape)
        # mip_arr_vstack = 
        # print()
        # quit()
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        # img_arg_npy_path = mip_pred_train_arg + img_name[:-7] + '.npy'
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        # arg_arr = np.load(img_arg_npy_path)


        arg_arr = np.argmax(img_arr, 0)
        print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
        # x = sitk.GetArrayFromImage(img)
        # y = sitk.GetArrayFromImage(lbl)
        mip_arr = np.max(img_arr, 0)
        print('mip_arr', mip_arr.shape, np.unique(img_arr))
        # quit()
        right_mip_label_arr = np.zeros_like(arg_arr)
        for i in range(0, arg_arr.shape[0]):
            for j in range(0, arg_arr.shape[1]):
                right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                
        right_mip_label_arr = np.expand_dims(right_mip_label_arr, 0)
        right_mip_label_arr = right_mip_label_arr.astype(np.int8)
        right_mip_label = sitk.GetImageFromArray(right_mip_label_arr)
        right_mip_label.SetDirection(direc)
        right_mip_label.SetOrigin(origin)
        right_mip_label.SetSpacing(spac)
        sitk.WriteImage(right_mip_label, '/memory/shanwenqi/datasets/nnUNet_raw_data/nnUNet_raw_data/Task815_MIP/Tr_lbl_3d_sparse_right/IXI035-IOP-0873-MRA-miplabel.nii.gz')
        
        mip_label_arr = np.max(lbl_arr, 0)
        print('mip_label_arr', mip_label_arr.shape, np.unique(mip_label_arr))
        # quit()
        
        # x = np.swapaxes(x, 0,2)
        # mip_to_sparse_label = np.zeros_like(x)
        sparse_3d_arr = np.zeros_like(img_arr)
        print('sparse_3d_arr', sparse_3d_arr.shape)
        # for num in range(0, x.shape[0]):
        # num = x.shape[0]
        # print('num', num)
        
        for i in range(arg_arr.shape[0]):
            for j in range(arg_arr.shape[1]):
                sparse_3d_arr[arg_arr[i][j], i, j] = right_mip_label_arr[0][i][j]
                # sparse_3d_arr[arg_arr[i][j], i, j] = lbl_arr[arg_arr[i][j], i, j]
                
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr)
        sparse_3d.CopyInformation(img)
        
        sitk.WriteImage(sparse_3d , save_path + img_name[:-7]+'sparse_3d_new_new22.nii.gz')
        quit()
        
        sum_slice = np.zeros((1024,1024))
        for i in range(0, 92):
            sum_slice += sparse_3d_arr[i,:,:]
        print('sum_slice', np.unique(sum_slice))
        xinagjian = sum_slice - mip_label_arr
        print('xinagjian', np.unique(xinagjian))
        print('xinagjian', sum_slice==mip_label_arr, np.unique(sum_slice==mip_label_arr))
        
        quit()
                
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr)
        sparse_3d.CopyInformation(img)
        
        sitk.WriteImage(sparse_3d , save_path + img_name[:-7]+'sparse_3d_new.nii.gz')
        
        quit()
        # argmax_bool_feat = (np.arange(num) == arg_arr[None,...])
        # print(argmax_bool_feat)
        print('argmax_bool_feat1', num, argmax_bool_feat.shape)
            
        aa = np.where(arg_arr==9)
        print(aa, len(aa[0]))
        bb = np.where(argmax_bool_feat[9]==True)
        print(bb, len(bb[0]))
        # print(aa==bb)
        # argmax_bool_feat = argmax_bool_feat
        # quit()
        argmax_bool_feat_arr = argmax_bool_feat
        argmax_bool_feat = argmax_bool_feat.astype(np.int8)
        # argmax_bool_feat = np.swapaxes(argmax_bool_feat, 0,2)
        print(argmax_bool_feat.shape,)
        # sparse_3d_arr = np.swapaxes(sparse_3d_arr, 0,2)
        argmax_bool_feat = sitk.GetImageFromArray(argmax_bool_feat)
        argmax_bool_feat.CopyInformation(img)
        sitk.WriteImage(argmax_bool_feat , save_path + img_name[:-7]+'args.nii.gz')
        
        
        # quit()
        sparse_3d_arr = np.multiply(mip_arr, argmax_bool_feat_arr) 
        
        # print(np.unique(sparse_3d_arr),)
        # cc = np.where(sparse_3d_arr[:,:,9]!=0)
        # print(cc, len(cc[0]))
        
        # sparse_3d_arr = np.swapaxes(sparse_3d_arr, 0,2)
        sparce_3d_lbl = sitk.GetImageFromArray(sparse_3d_arr)
        sparce_3d_lbl.CopyInformation(img)
        sitk.WriteImage(sparce_3d_lbl , save_path + img_name)
        
        print()
        # quit()
        # depth = 64 
        # argmax_bool = np.zeros((192,192,64))
        
        # mip_index_all = np.load('/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_path/IXI035-IOP-0873-MRA_0.npy')
        # mip_index = mip_index_all[192:384, 192:384]
        # print('mip_index', mip_index.shape, np.unique(mip_index))
        
        # # mip_index = np.expand_dims(mip_index, axis = 0)
        # # mip_index = np.expand_dims(mip_index, axis = 0)
        # # print('mip_index', mip_index.shape)
        
        # argmax_bool = (np.arange(depth) == mip_index[...,None])
        # print('argmax_bool', argmax_bool.shape, np.unique(argmax_bool))
        
        # index_1 = np.where(mip_index==5)
        # print(index_1, len(index_1[0]))
        
        # argmax_bool_4 = argmax_bool[:,:,5]
        # index_4 = np.where(argmax_bool_4==True)
        # print(index_4, len(index_4[0]))
        
        # quit()
       

def right_mip_pred_and_args_to_3d_sparse_seg():
    save_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_pred_3d_sparse_right/'
    os.makedirs(save_path, exist_ok=True)
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/Noskull_N4_bias/'
    # img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image/'
    
    for img_name in os.listdir(img_path):
        print(img_name)
        img = sitk.ReadImage(img_path + img_name)
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        # mip_pred = sitk.ReadImage(mip_pred_train_path + img_name)
        #mip-pred转化为3dsparse
        mip_pred_train_path = '/memory/shanwenqi/datasets/nnUNet_raw_data/nnUNet_raw_data/Task815_MIP/Tspred/'
        
        right_mip_label_img = sitk.ReadImage(mip_pred_train_path + img_name)
        right_mip_label_arr = sitk.GetArrayFromImage(right_mip_label_img)
        right_mip_label_arr[right_mip_label_arr==2]=1
        print('right_mip_label_arr', np.unique(right_mip_label_arr))
        
        arg_arr = np.argmax(img_arr, 0)
        '''正确的生成 mip-label 的过程
        print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
        
        right_mip_label_arr = np.zeros_like(arg_arr)
        for i in range(0, arg_arr.shape[0]):
            for j in range(0, arg_arr.shape[1]):
                right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                
        right_mip_label_arr = np.expand_dims(right_mip_label_arr, 0)
        right_mip_label_arr = right_mip_label_arr.astype(np.int8)
        right_mip_label = sitk.GetImageFromArray(right_mip_label_arr)
        right_mip_label.SetDirection(direc)
        right_mip_label.SetOrigin(origin)
        right_mip_label.SetSpacing(spac)
        mip_lbl_train_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_allslice/'
        sitk.WriteImage(right_mip_label, mip_lbl_train_path + img_name)
        '''
        sparse_3d_arr = np.zeros_like(img_arr)
        print('sparse_3d_arr', sparse_3d_arr.shape)
        
        for i in range(arg_arr.shape[0]):
            for j in range(arg_arr.shape[1]):
                sparse_3d_arr[arg_arr[i][j], i, j] = right_mip_label_arr[0][i][j]
                # sparse_3d_arr[arg_arr[i][j], i, j] = lbl_arr[arg_arr[i][j], i, j]
                
        xiangjian = lbl_arr - sparse_3d_arr
        print(np.unique(xiangjian))
        # assert -1 not in np.unique(xiangjian); "稀疏标签有噪声错误！"
        # quit()
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr)
        sparse_3d.CopyInformation(img)
        
        sitk.WriteImage(sparse_3d , save_path + img_name)
        # quit()



def right_25D_mip_pred_and_args_to_3d_sparse_seg():
    save_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/mip_pred_25d_sparse_right/'
    os.makedirs(save_path, exist_ok=True)
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/Noskull_N4_bias/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/label_no_skull/'
    
    for img_name in os.listdir(img_path):
        print(img_name)
        img = sitk.ReadImage(img_path + img_name)
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        # mip_pred = sitk.ReadImage(mip_pred_train_path + img_name)
        #mip-pred转化为3dsparse
        mip_pred_train_path = '/memory/shanwenqi/datasets/nnUNet_trained_models/nnUNet/3d_fullres/Task817_IXI45_MIP_2.5D/nnUNetTrainerV2__nnUNetPlansv2.1/all/validation_raw/'
        # mip_pred_train_path = '//memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/mip_lbl_2.5d_allslice/'
        sparse_3d_arr_end =  np.zeros_like(img_arr)
        for ax in [0,1,2]:
            img_name_ax = img_name[:-7] + 'axis_%s.nii.gz'%ax
            
            right_mip_label_img = sitk.ReadImage(mip_pred_train_path + img_name_ax)
            right_mip_label_arr = sitk.GetArrayFromImage(right_mip_label_img)
            right_mip_label_arr[right_mip_label_arr==2]=1
            print('right_mip_label_arr', np.unique(right_mip_label_arr))
        
            arg_arr = np.argmax(img_arr, ax)
            '''正确的生成 mip-label 的过程
            print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
            
            right_mip_label_arr = np.zeros_like(arg_arr)
            for i in range(0, arg_arr.shape[0]):
                for j in range(0, arg_arr.shape[1]):
                    right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                    
            right_mip_label_arr = np.expand_dims(right_mip_label_arr, 0)
            right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            right_mip_label = sitk.GetImageFromArray(right_mip_label_arr)
            right_mip_label.SetDirection(direc)
            right_mip_label.SetOrigin(origin)
            right_mip_label.SetSpacing(spac)
            mip_lbl_train_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_allslice/'
            sitk.WriteImage(right_mip_label, mip_lbl_train_path + img_name)
            '''
            sparse_3d_arr = np.zeros_like(img_arr)
            print('sparse_3d_arr', sparse_3d_arr.shape)
            
            for i in range(arg_arr.shape[0]):
                for j in range(arg_arr.shape[1]):
                    if ax==0:
                        sparse_3d_arr[arg_arr[i][j], i, j] = right_mip_label_arr[0][i][j]
                    elif ax==1:
                        sparse_3d_arr[i, arg_arr[i][j], j] = right_mip_label_arr[0][i][j]
                    elif ax==2:
                        sparse_3d_arr[i, j, arg_arr[i][j]] = right_mip_label_arr[0][i][j]
                # sparse_3d_arr[arg_arr[i][j], i, j] = lbl_arr[arg_arr[i][j], i, j]
            sparse_3d_arr_end = np.logical_or(sparse_3d_arr_end, sparse_3d_arr)
            
        xiangjian = lbl_arr - sparse_3d_arr_end
        print(np.unique(sparse_3d_arr_end))
        # assert -1 not in np.unique(xiangjian); "稀疏标签有噪声错误！"
        # quit()
        sparse_3d_arr_end = sparse_3d_arr_end.astype(np.int8)
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr_end)
        sparse_3d.CopyInformation(img)
        print(save_path + img_name)
        # sitk.WriteImage(sparse_3d , save_path + img_name)
        # quit()
        
        
# right_25D_mip_pred_and_args_to_3d_sparse_seg()




def right_25D_mip_pred_and_args_to_3d_sparse_seg_IXI_HH():
    save_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/mip_lbl_25d_sparse_right/'
    os.makedirs(save_path, exist_ok=True)
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/all_img/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/all_lbl/'
    
    for img_name in os.listdir(img_path):
        print(img_name)
        img = sitk.ReadImage(img_path + img_name)
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        # mip_pred = sitk.ReadImage(mip_pred_train_path + img_name)
        #mip-pred转化为3dsparse
        # mip_pred_train_path = '/memory/shanwenqi/datasets/nnUNet_trained_models/nnUNet/3d_fullres/Task817_IXI45_MIP_2.5D/nnUNetTrainerV2__nnUNetPlansv2.1/all/validation_raw/'
        mip_pred_train_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/mip_lbl_2.5d_allslice/'
        sparse_3d_arr_end =  np.zeros_like(img_arr)
        for ax in [0,1,2]:
            img_name_ax = img_name[:-7] + 'axis_%s_lbl_slice_all.nii.gz'%ax
            
            right_mip_label_img = sitk.ReadImage(mip_pred_train_path + img_name_ax)
            right_mip_label_arr = sitk.GetArrayFromImage(right_mip_label_img)
            # right_mip_label_arr[right_mip_label_arr==2]=1
            print('right_mip_label_arr', np.unique(right_mip_label_arr))
            # quit()
            arg_arr = np.argmax(img_arr, ax)
            '''正确的生成 mip-label 的过程
            print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
            
            right_mip_label_arr = np.zeros_like(arg_arr)
            for i in range(0, arg_arr.shape[0]):
                for j in range(0, arg_arr.shape[1]):
                    right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                    
            right_mip_label_arr = np.expand_dims(right_mip_label_arr, 0)
            right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            right_mip_label = sitk.GetImageFromArray(right_mip_label_arr)
            right_mip_label.SetDirection(direc)
            right_mip_label.SetOrigin(origin)
            right_mip_label.SetSpacing(spac)
            mip_lbl_train_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_allslice/'
            sitk.WriteImage(right_mip_label, mip_lbl_train_path + img_name)
            '''
            sparse_3d_arr = np.zeros_like(img_arr)
            print('sparse_3d_arr', sparse_3d_arr.shape)
            
            for i in range(arg_arr.shape[0]):
                for j in range(arg_arr.shape[1]):
                    if ax==0:
                        sparse_3d_arr[arg_arr[i][j], i, j] = right_mip_label_arr[0][i][j]
                    elif ax==1:
                        sparse_3d_arr[i, arg_arr[i][j], j] = right_mip_label_arr[0][i][j]
                    elif ax==2:
                        sparse_3d_arr[i, j, arg_arr[i][j]] = right_mip_label_arr[0][i][j]
                # sparse_3d_arr[arg_arr[i][j], i, j] = lbl_arr[arg_arr[i][j], i, j]
            sparse_3d_arr_end = np.logical_or(sparse_3d_arr_end, sparse_3d_arr)
            
        xiangjian = lbl_arr - sparse_3d_arr_end
        print(np.unique(sparse_3d_arr_end))
        assert -1 not in np.unique(xiangjian); "稀疏标签有噪声错误！"
        # quit()
        sparse_3d_arr_end = sparse_3d_arr_end.astype(np.int8)
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr_end)
        sparse_3d.CopyInformation(img)
        print(save_path + img_name)
        sitk.WriteImage(sparse_3d , save_path + img_name)
        # quit()
        
        
# right_25D_mip_pred_and_args_to_3d_sparse_seg_IXI_HH()


def right_25D_mip_pred_and_args_to_3d_sparse_seg_ADAM():
    save_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/mip_lbl_25d_sparse_right/'
    os.makedirs(save_path, exist_ok=True)
    img_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/all_img_align/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/all_lbl_align/'
    
    for img_name in os.listdir(img_path):
        print(img_name)
        img = sitk.ReadImage(img_path + img_name)
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        # mip_pred = sitk.ReadImage(mip_pred_train_path + img_name)
        #mip-pred转化为3dsparse
        # mip_pred_train_path = '/memory/shanwenqi/datasets/nnUNet_trained_models/nnUNet/3d_fullres/Task817_IXI45_MIP_2.5D/nnUNetTrainerV2__nnUNetPlansv2.1/all/validation_raw/'
        mip_pred_train_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/mip_lbl_2.5d_allslice/'
        sparse_3d_arr_end =  np.zeros_like(img_arr)
        for ax in [0,1,2]:
            img_name_ax = img_name[:-7] + 'axis_%s_lbl_slice_all.nii.gz'%ax
            
            right_mip_label_img = sitk.ReadImage(mip_pred_train_path + img_name_ax)
            right_mip_label_arr = sitk.GetArrayFromImage(right_mip_label_img)
            # right_mip_label_arr[right_mip_label_arr==2]=1
            print('right_mip_label_arr', np.unique(right_mip_label_arr))
            # quit()
            arg_arr = np.argmax(img_arr, ax)
            '''正确的生成 mip-label 的过程
            print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
            
            right_mip_label_arr = np.zeros_like(arg_arr)
            for i in range(0, arg_arr.shape[0]):
                for j in range(0, arg_arr.shape[1]):
                    right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                    
            right_mip_label_arr = np.expand_dims(right_mip_label_arr, 0)
            right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            right_mip_label = sitk.GetImageFromArray(right_mip_label_arr)
            right_mip_label.SetDirection(direc)
            right_mip_label.SetOrigin(origin)
            right_mip_label.SetSpacing(spac)
            mip_lbl_train_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_allslice/'
            sitk.WriteImage(right_mip_label, mip_lbl_train_path + img_name)
            '''
            sparse_3d_arr = np.zeros_like(img_arr)
            print('sparse_3d_arr', sparse_3d_arr.shape)
            
            for i in range(arg_arr.shape[0]):
                for j in range(arg_arr.shape[1]):
                    if ax==0:
                        sparse_3d_arr[arg_arr[i][j], i, j] = right_mip_label_arr[0][i][j]
                    elif ax==1:
                        sparse_3d_arr[i, arg_arr[i][j], j] = right_mip_label_arr[0][i][j]
                    elif ax==2:
                        sparse_3d_arr[i, j, arg_arr[i][j]] = right_mip_label_arr[0][i][j]
                # sparse_3d_arr[arg_arr[i][j], i, j] = lbl_arr[arg_arr[i][j], i, j]
            sparse_3d_arr_end = np.logical_or(sparse_3d_arr_end, sparse_3d_arr)
            
        xiangjian = lbl_arr - sparse_3d_arr_end
        print(np.unique(sparse_3d_arr_end))
        assert -1 not in np.unique(xiangjian); "稀疏标签有噪声错误！"
        # quit()
        sparse_3d_arr_end = sparse_3d_arr_end.astype(np.int8)
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr_end)
        sparse_3d.CopyInformation(img)
        print(save_path + img_name)
        sitk.WriteImage(sparse_3d , save_path + img_name)
        # quit()
        
        
right_25D_mip_pred_and_args_to_3d_sparse_seg_ADAM()