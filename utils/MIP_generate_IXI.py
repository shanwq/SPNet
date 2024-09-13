import numpy as np
import SimpleITK as sitk
import os

def mip_generate_all_IXI_HH():
    
    img_path = './all_img/'
    lbl_path = './all_lbl/'
    mip_img_allslice_path = './mip_img_2.5d_allslice/'
    mip_lbl_allslice_path = './mip_lbl_2.5d_allslice/'

    os.makedirs(mip_img_allslice_path, exist_ok=True)
    os.makedirs(mip_lbl_allslice_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        
        # img_npy_path = mip_img_16slice_path + img_name[:-7]
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        x = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        
        for ax in [0,1,2]:
            
            mip_x_img = np.max(x, ax)
            print(mip_x_img.shape)
            mip_img_arr = np.expand_dims(mip_x_img, axis=0)
                
            arg_arr = np.argmax(x, ax)
            print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
            # quit()
            right_mip_label_arr = np.zeros_like(arg_arr)
            for i in range(0, arg_arr.shape[0]):
                for j in range(0, arg_arr.shape[1]):
                    if ax==0:
                        right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                    elif ax==1:
                        right_mip_label_arr[i,j] = lbl_arr[i, arg_arr[i][j],  j]
                    elif ax==2:
                        right_mip_label_arr[i,j] = lbl_arr[i, j, arg_arr[i][j]]
                        
                    
            right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            mip_lbl_arr = np.expand_dims(right_mip_label_arr, axis=0)
            
            mip_img = sitk.GetImageFromArray(mip_img_arr)
            mip_lbl = sitk.GetImageFromArray(mip_lbl_arr)
            
            mip_img.SetDirection(direc)
            mip_lbl.SetDirection(direc)
            mip_img.SetOrigin(origin)
            mip_lbl.SetOrigin(origin)
            mip_img.SetSpacing(spac)
            mip_lbl.SetSpacing(spac)
            
            
            sitk.WriteImage(mip_img, os.path.join(mip_img_allslice_path, '%s'%(img_name[:-7]) + 'axis_%s_img_slice_all.nii.gz'%ax))
            sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_allslice_path, '%s'%(img_name[:-7]) + 'axis_%s_lbl_slice_all.nii.gz'%ax))

def mip_generate_all_ADAM():
    
    img_path = './all_img_align/'
    lbl_path = './all_lbl_align/'
    mip_img_allslice_path = './mip_img_2.5d_allslice/'
    mip_lbl_allslice_path = './mip_lbl_2.5d_allslice/'

    os.makedirs(mip_img_allslice_path, exist_ok=True)
    os.makedirs(mip_lbl_allslice_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        
        # img_npy_path = mip_img_16slice_path + img_name[:-7]
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        
        x = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)
        
        for ax in [0,1,2]:
            
            mip_x_img = np.max(x, ax)
            print(mip_x_img.shape)
            mip_img_arr = np.expand_dims(mip_x_img, axis=0)
                
            arg_arr = np.argmax(x, ax)
            print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
            # quit()
            right_mip_label_arr = np.zeros_like(arg_arr)
            for i in range(0, arg_arr.shape[0]):
                for j in range(0, arg_arr.shape[1]):
                    if ax==0:
                        right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
                    elif ax==1:
                        right_mip_label_arr[i,j] = lbl_arr[i, arg_arr[i][j],  j]
                    elif ax==2:
                        right_mip_label_arr[i,j] = lbl_arr[i, j, arg_arr[i][j]]
                        
                    
            right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            mip_lbl_arr = np.expand_dims(right_mip_label_arr, axis=0)
            
            mip_img = sitk.GetImageFromArray(mip_img_arr)
            mip_lbl = sitk.GetImageFromArray(mip_lbl_arr)
            # mip_img.CopyInformation(img)
            # mip_lbl.CopyInformation(img)
            
            mip_img.SetDirection(direc)
            mip_lbl.SetDirection(direc)
            mip_img.SetOrigin(origin)
            mip_lbl.SetOrigin(origin)
            mip_img.SetSpacing(spac)
            mip_lbl.SetSpacing(spac)
            
            
            sitk.WriteImage(mip_img, os.path.join(mip_img_allslice_path, '%s'%(img_name[:-7]) + 'axis_%s_img_slice_all.nii.gz'%ax))
            sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_allslice_path, '%s'%(img_name[:-7]) + 'axis_%s_lbl_slice_all.nii.gz'%ax))
if __name__ == '__main__':
    mip_generate_all_ADAM()
    mip_generate_all_IXI_HH()


