import numpy as np
import SimpleITK as sitk
import os

img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image/'
lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_label/'
mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_img_64slice/'
mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_64slice/'

# os.makedirs(mip_img_16slice_path, exist_ok=True)
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
mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_img_allslice/'
mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_allslice/'

# os.makedirs(mip_img_allslice_path, exist_ok=True)
# os.makedirs(mip_lbl_allslice_path, exist_ok=True)
def mip_generate_all():
    
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/Noskull_N4_bias/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/label_no_skull/'
    mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/mip_img_2.5d_allslice/'
    mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/all/mip_lbl_2.5d_allslice/'

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
                
# mip_generate_all()
# mip_generate()

def mip_generate_all_IXI_HH():
    
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/all_img/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/all_lbl/'
    mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/mip_img_2.5d_allslice/'
    mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_HH/mip_lbl_2.5d_allslice/'

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
                
# mip_generate_all_IXI_HH()


def mip_generate_all_ADAM():
    
    img_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/all_img_align/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/all_lbl_align/'
    mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/mip_img_2.5d_allslice/'
    mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/Vessel_ADAM/mip_lbl_2.5d_allslice/'

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
                
mip_generate_all_ADAM()


quit()



import numpy as np
import SimpleITK as sitk
import os



'''我自己的重叠式 MIP 生成过程'''
def mip_generate(n_slice):
    n_slice = n_slice
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/noskull_N4_image/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/noskull_N4_label/'
    mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_img_%s_slice_3D/'%n_slice
    mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_%s_slice_3D/'%n_slice
    mip_arg_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_arg_%s_slice_3D/'%n_slice

    os.makedirs(mip_img_16slice_path, exist_ok=True)
    os.makedirs(mip_lbl_16slice_path, exist_ok=True)
    os.makedirs(mip_arg_16slice_path, exist_ok=True)
    
    for img_name in os.listdir(img_path):
        
        img = sitk.ReadImage(img_path + img_name)
        lbl = sitk.ReadImage(lbl_path + img_name)
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
        img_arr = sitk.GetArrayFromImage(img)
        lbl_arr = sitk.GetArrayFromImage(lbl)

        mip_img_all = np.zeros_like(img_arr)
        mip_lbl_all = np.zeros_like(img_arr)
        mip_arg_all = np.zeros_like(img_arr)
        
        for kk in range(0, img_arr.shape[0]):
            if kk < int((n_slice-1)/2) :
                print(kk)
                img_patch_arr = img_arr[:n_slice,:,:]
                lbl_patch_arr = lbl_arr[:n_slice,:,:]
                mip_img_n = np.max(img_patch_arr, 0)
                arg_arr = np.argmax(img_patch_arr, 0)
                print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
                right_mip_label_arr = np.zeros_like(arg_arr)
                for i in range(0, arg_arr.shape[0]):
                    for j in range(0, arg_arr.shape[1]):
                        right_mip_label_arr[i,j] = lbl_patch_arr[arg_arr[i][j], i, j]
                right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            elif kk > ((img_arr.shape[0] - int((n_slice-1)/2))-1):
                print(kk)
                img_patch_arr = img_arr[-1*n_slice:,:,:]
                lbl_patch_arr = lbl_arr[-1*n_slice:,:,:]
                mip_img_n = np.max(img_patch_arr, 0)
                arg_arr = np.argmax(img_patch_arr, 0)
                print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
                right_mip_label_arr = np.zeros_like(arg_arr)
                for i in range(0, arg_arr.shape[0]):
                    for j in range(0, arg_arr.shape[1]):
                        right_mip_label_arr[i,j] = lbl_patch_arr[arg_arr[i][j], i, j]
                right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            else:
                img_patch_arr = img_arr[kk-int((n_slice-1)/2): kk+int((n_slice-1)/2)+1]
                lbl_patch_arr = lbl_arr[kk-int((n_slice-1)/2): kk+int((n_slice-1)/2)+1]
                mip_img_n = np.max(img_patch_arr, 0)
                arg_arr = np.argmax(img_patch_arr, 0)
                print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
                right_mip_label_arr = np.zeros_like(arg_arr)
                for i in range(0, arg_arr.shape[0]):
                    for j in range(0, arg_arr.shape[1]):
                        right_mip_label_arr[i,j] = lbl_patch_arr[arg_arr[i][j], i, j]
                right_mip_label_arr = right_mip_label_arr.astype(np.int8)
            
            mip_img_all[kk,:,:] = mip_img_n
            mip_lbl_all[kk,:,:] = right_mip_label_arr
            mip_arg_all[kk,:,:] = arg_arr
            
        mip_img = sitk.GetImageFromArray(mip_img_all)
        mip_lbl = sitk.GetImageFromArray(mip_lbl_all)
        mip_arg = sitk.GetImageFromArray(mip_arg_all)
        mip_img.CopyInformation(img)
        mip_lbl.CopyInformation(img)
        mip_arg.CopyInformation(img)
        # mip_img.SetDirection(direc)
        # mip_lbl.SetDirection(direc)
        # mip_img.SetOrigin(origin)
        # mip_lbl.SetOrigin(origin)
        # mip_img.SetSpacing(spac)
        # mip_lbl.SetSpacing(spac)
            
        sitk.WriteImage(mip_img, os.path.join(mip_img_16slice_path, img_name))#'%s'%(img_name[:-7]) + 'img_slice_%s.nii.gz'%n_))
        sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_16slice_path, img_name))#'%s'%(img_name[:-7]) + 'lbl_slice_%s.nii.gz'%kk))
        sitk.WriteImage(mip_arg, os.path.join(mip_arg_16slice_path, img_name))#'%s'%(img_name[:-7]) + 'lbl_slice_%s.nii.gz'%kk))
        
            # mip_img_all[kk,:,:] = mip_x_img
            # mip_lbl_all[kk,:,:] = mip_y_img
            # np.save(os.path.join(mip_img_16slice_path, '%s'%(img_name[:-7]) + 'img_slice_%s.npy'%kk), mip_x_img)
            # np.save(os.path.join(mip_lbl_16slice_path, '%s'%(img_name[:-7]) + 'lbl_slice_%s.npy'%kk), mip_y_img)
n_slice = 15
mip_generate(n_slice)

mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_img_allslice/'
mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/mip_lbl_allslice/'

os.makedirs(mip_img_allslice_path, exist_ok=True)
os.makedirs(mip_lbl_allslice_path, exist_ok=True)
def mip_generate_all():
    img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image/'
    lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_label/'
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

        mip_x_img = np.max(x, 0)
        print(mip_x_img.shape)
        mip_img_arr = np.expand_dims(mip_x_img, axis=0)
            
        arg_arr = np.argmax(x, 0)
        print('arg_arr.shape', arg_arr.shape, np.unique(arg_arr))
        # quit()
        right_mip_label_arr = np.zeros_like(arg_arr)
        for i in range(0, arg_arr.shape[0]):
            for j in range(0, arg_arr.shape[1]):
                right_mip_label_arr[i,j] = lbl_arr[arg_arr[i][j], i, j]
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
        
        
        sitk.WriteImage(mip_img, os.path.join(mip_img_allslice_path, '%s'%(img_name[:-7]) + 'img_slice_all.nii.gz'))
        sitk.WriteImage(mip_lbl, os.path.join(mip_lbl_allslice_path, '%s'%(img_name[:-7]) + 'lbl_slice_all.nii.gz'))
            
mip_generate_all()
# mip_generate()



# img_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_image/'
# lbl_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/noskull_N4_label/'
# mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/img_slice/'
# mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/test/seed_3047/lbl_slice/'

# os.makedirs(mip_img_16slice_path, exist_ok=True)
# os.makedirs(mip_lbl_16slice_path, exist_ok=True)

# for img_name in os.listdir(img_path):
    
#     img = sitk.ReadImage(img_path + img_name)
#     lbl = sitk.ReadImage(lbl_path + img_name)

#     spac = img.GetSpacing()
#     origin = img.GetOrigin()
#     direc = img.GetDirection()
    
#     x = sitk.GetArrayFromImage(img)
#     y = sitk.GetArrayFromImage(lbl)
#     print(x.shape)
#     # quit()
#     npy_img_path = mip_img_16slice_path + img_name[:-7]
#     os.makedirs(npy_img_path, exist_ok=True)
    
#     npy_lbl_path = mip_lbl_16slice_path + img_name[:-7]
#     os.makedirs(npy_lbl_path, exist_ok=True)
    
#     for kk in range(len(x)):
#         npy_slice_img = x[kk,:,:]
#         rgb_slice_img = np.zeros((3, 1024, 1024))
#         rgb_slice_img[0] = npy_slice_img
#         rgb_slice_img[1] = npy_slice_img
#         rgb_slice_img[2] = npy_slice_img
#         np.save(os.path.join(npy_img_path, 'slice_%s'%str(kk)) , rgb_slice_img)
        
#         npy_slice_lbl = y[kk,:,:]
#         rgb_slice_lbl = np.zeros((3, 1024, 1024))
#         rgb_slice_lbl[0] = npy_slice_lbl
#         rgb_slice_lbl[1] = npy_slice_lbl
#         rgb_slice_lbl[2] = npy_slice_lbl
#         np.save(os.path.join(npy_lbl_path, 'slice_%s'%str(kk)) , rgb_slice_lbl)
    
#     quit()
    # mip_img = np.max(x, 0)
    # mip_img = np.expand_dims(mip_img, axis=0)
    # mip_lbl = np.max(y, 0)
    # mip_lbl = np.expand_dims(mip_lbl, axis=0)
    # print('mip_img',mip_img.shape)
    # print('mip_lbl',mip_lbl.shape)
    # # quit()
    # mip_img = sitk.GetImageFromArray(mip_img)
    # mip_lbl = sitk.GetImageFromArray(mip_lbl)
    # # mip_img.CopyInformation(img)
    # # mip_lbl.CopyInformation(img)
    
    # mip_img.SetDirection(direc)
    # mip_lbl.SetDirection(direc)
    # mip_img.SetOrigin(origin)
    # mip_lbl.SetOrigin(origin)
    # mip_img.SetSpacing(spac)
    # mip_lbl.SetSpacing(spac)
    
    
    # sitk.WriteImage(mip_img, mip_img_16slice_path + img_name)
    # sitk.WriteImage(mip_lbl, mip_lbl_16slice_path + img_name)
    