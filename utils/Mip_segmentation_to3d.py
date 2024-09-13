import numpy as np
import SimpleITK as sitk
import os

def right_25D_mip_pred_and_args_to_3d_sparse_seg_IXI_HH():
    save_path = './mip_lbl_25d_sparse_mask/'
    os.makedirs(save_path, exist_ok=True)
    img_path = './all_img/'
    lbl_path = './all_lbl/'
    
    for img_name in os.listdir(img_path):
        print(img_name)
        img = sitk.ReadImage(img_path + img_name)
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        lbl = sitk.ReadImage(lbl_path + img_name)
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        #mip-pred转化为3dsparse
        mip_pred_train_path = './mip_pred_2.5d_allslice/'
        sparse_3d_arr_end =  np.zeros_like(img_arr)
        for ax in [0,1,2]:
            img_name_ax = img_name[:-7] + 'axis_%s_lbl_slice_all.nii.gz'%ax
            
            right_mip_label_img = sitk.ReadImage(mip_pred_train_path + img_name_ax)
            right_mip_label_arr = sitk.GetArrayFromImage(right_mip_label_img)
            # right_mip_label_arr[right_mip_label_arr==2]=1
            print('right_mip_label_arr', np.unique(right_mip_label_arr))
            # quit()
            arg_arr = np.argmax(img_arr, ax)

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
            sparse_3d_arr_end = np.logical_or(sparse_3d_arr_end, sparse_3d_arr)
            
        sparse_3d_arr_end = sparse_3d_arr_end.astype(np.int8)
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr_end)
        sparse_3d.CopyInformation(img)
        print(save_path + img_name)
        sitk.WriteImage(sparse_3d , save_path + img_name)

def right_25D_mip_pred_and_args_to_3d_sparse_seg_ADAM():
    save_path = './Vessel_ADAM/mip_pred_25d_sparse_mask/'
    os.makedirs(save_path, exist_ok=True)
    img_path = './Vessel_ADAM/all_img_align/'
    lbl_path = './Vessel_ADAM/all_lbl_align/'
    
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
        mip_pred_train_path = './Vessel_ADAM/mip_pred_2.5d_allslice/'
        sparse_3d_arr_end =  np.zeros_like(img_arr)
        for ax in [0,1,2]:
            img_name_ax = img_name[:-7] + 'axis_%s_lbl_slice_all.nii.gz'%ax
            
            right_mip_label_img = sitk.ReadImage(mip_pred_train_path + img_name_ax)
            right_mip_label_arr = sitk.GetArrayFromImage(right_mip_label_img)
            # right_mip_label_arr[right_mip_label_arr==2]=1
            print('right_mip_label_arr', np.unique(right_mip_label_arr))
            arg_arr = np.argmax(img_arr, ax)
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
            sparse_3d_arr_end = np.logical_or(sparse_3d_arr_end, sparse_3d_arr)
            
        print(np.unique(sparse_3d_arr_end))
        sparse_3d_arr_end = sparse_3d_arr_end.astype(np.int8)
        sparse_3d = sitk.GetImageFromArray(sparse_3d_arr_end)
        sparse_3d.CopyInformation(img)
        print(save_path + img_name)
        sitk.WriteImage(sparse_3d , save_path + img_name)

if __name__ = '__main':
    
    right_25D_mip_pred_and_args_to_3d_sparse_seg_ADAM()
    right_25D_mip_pred_and_args_to_3d_sparse_seg_IXI_HH()
    
