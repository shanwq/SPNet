import numpy as np
import SimpleITK as sitk
import os

def mip_generate():
    mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_imagesTr/'
    mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_labelsTr/'
    for data_name in os.listdir(img_path):
        dataset_image_path = img_path + data_name + '/'
        dataset_label_path = lbl_path + data_name + '/'
        
        for img_name in os.listdir(dataset_image_path):
            img_path = dataset_image_path + img_name
            lbl_path = dataset_label_path + img_name
            
            img = sitk.ReadImage(img_path)# + img_name)
            lbl = sitk.ReadImage(lbl_path)# + img_name)

    
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
# mip_img_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_img_allslice/'
# mip_lbl_allslice_path = '/memory/shanwenqi/Vessel_seg/IXI_MRA/45_cases/train/seed_3047/mip_lbl_allslice/'

# os.makedirs(mip_img_allslice_path, exist_ok=True)
# os.makedirs(mip_lbl_allslice_path, exist_ok=True)
def mip_generate_all_train():
    img_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/imagesTr/'
    lbl_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/labelsTr/'
    mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_imagesTr/'
    mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_labelsTr/'
    os.makedirs(mip_img_16slice_path, exist_ok=True)
    os.makedirs(mip_lbl_16slice_path, exist_ok=True)
    for data_name in os.listdir(img_path_all):
        dataset_image_path = img_path_all + data_name + '/'
        dataset_label_path = lbl_path_all + data_name + '/'
        
        for img_name in os.listdir(dataset_image_path):
            img_path = dataset_image_path + img_name
            lbl_path = dataset_label_path + img_name
            img = sitk.ReadImage(img_path)# + img_name)
            lbl = sitk.ReadImage(lbl_path)# + img_name)
        # img_npy_path = mip_img_16slice_path + img_name[:-7]
        # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
        
            spac = img.GetSpacing()
            origin = img.GetOrigin()
            direc = img.GetDirection()
        
            x = sitk.GetArrayFromImage(img)
            # print(x.shape)
            mip_x_img = np.max(x, 0)
            
            lbl_arr = sitk.GetArrayFromImage(lbl)
            # mip_pred = sitk.ReadImage(mip_pred_train_path + img_name)
            print(mip_x_img.shape)
            # quit()
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
            print('mip_img',mip_img_arr.shape)
            print('mip_lbl',mip_lbl_arr.shape)
            # quit()
            mip_img = sitk.GetImageFromArray(mip_img_arr)
            mip_lbl = sitk.GetImageFromArray(mip_lbl_arr)
            
            mip_img.SetDirection(direc)
            mip_lbl.SetDirection(direc)
            mip_img.SetOrigin(origin)
            mip_lbl.SetOrigin(origin)
            mip_img.SetSpacing(spac)
            mip_lbl.SetSpacing(spac)
            
            save_img_path = mip_img_16slice_path + data_name + '/'
            save_lbl_path = mip_lbl_16slice_path + data_name + '/'
            os.makedirs(save_img_path, exist_ok=True)
            os.makedirs(save_lbl_path, exist_ok=True)
            
            # print(save_img_path, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
            # print(save_lbl_path)
            
            sitk.WriteImage(mip_img, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
            sitk.WriteImage(mip_lbl, os.path.join(save_lbl_path, ('%s'%(img_name[:-7]) + 'lbl_slice_all.nii.gz')))
            # quit()
        
    
def mip_generate_all_test():
    img_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/imagesTs/'
    lbl_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/labelsTs/'
    mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_imagesTs/'
    mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_labelsTs/'
    os.makedirs(mip_img_16slice_path, exist_ok=True)
    os.makedirs(mip_lbl_16slice_path, exist_ok=True)

    
    for img_name in os.listdir(img_path_all):
        img_path = img_path_all + img_name
        lbl_path = lbl_path_all + img_name
        
        img = sitk.ReadImage(img_path)# + img_name)
        lbl = sitk.ReadImage(lbl_path)# + img_name)

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
        print('mip_img',mip_img_arr.shape)
        print('mip_lbl',mip_lbl_arr.shape)
        # quit()
        mip_img = sitk.GetImageFromArray(mip_img_arr)
        mip_lbl = sitk.GetImageFromArray(mip_lbl_arr)
    
        mip_img.SetDirection(direc)
        mip_lbl.SetDirection(direc)
        mip_img.SetOrigin(origin)
        mip_lbl.SetOrigin(origin)
        mip_img.SetSpacing(spac)
        mip_lbl.SetSpacing(spac)
        
        save_img_path = mip_img_16slice_path
        save_lbl_path = mip_lbl_16slice_path 
        os.makedirs(save_img_path, exist_ok=True)
        os.makedirs(save_lbl_path, exist_ok=True)
        
        # print(save_img_path, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
        # print(save_lbl_path)
        
        sitk.WriteImage(mip_img, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
        sitk.WriteImage(mip_lbl, os.path.join(save_lbl_path, ('%s'%(img_name[:-7]) + 'lbl_slice_all.nii.gz')))

def mip_generate_all_train_argdepth():
    img_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/imagesTr/'
    lbl_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/labelsTr/'
    mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_imagesTr/'
    mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_labelsTr/'
    os.makedirs(mip_img_16slice_path, exist_ok=True)
    os.makedirs(mip_lbl_16slice_path, exist_ok=True)
    for data_name in os.listdir(img_path_all):
        dataset_image_path = img_path_all + data_name + '/'
        dataset_label_path = lbl_path_all + data_name + '/'
        
        for img_name in os.listdir(dataset_image_path):
            img_path = dataset_image_path + img_name
            lbl_path = dataset_label_path + img_name
            
            img = sitk.ReadImage(img_path)# + img_name)
            lbl = sitk.ReadImage(lbl_path)# + img_name)

            spac = img.GetSpacing()
            origin = img.GetOrigin()
            direc = img.GetDirection()
        
            x = sitk.GetArrayFromImage(img)
            y = sitk.GetArrayFromImage(lbl)
                
            mip_x_img = np.max(x, 0)
            mip_y_img = np.max(y, 0)
            # print(mip_x_img.shape)
            
            
            
            
            mip_x_arg = np.argmax(x, 0)
            mip_y_arg = np.argmax(y, 0)
            
            print('mip_x_arg, mip_y_arg', x.shape[0], mip_x_arg, mip_y_arg)
            
            x_flip = x[::-1, :, :]
            y_flip = y[::-1, :, :]
            
            mip_flip_x_arg = np.argmax(x_flip, 0)
            mip_flip_y_arg = np.argmax(y_flip, 0)
            print('mip_flip_x_arg, mip_flip_y_arg',  x.shape[0], mip_flip_x_arg, mip_flip_y_arg)
            print('x.shape[0], mip_x_arg + mip_flip_x_arg',  x.shape[0], mip_x_arg + mip_flip_x_arg+87, )
            print(np.unique(mip_x_arg + mip_flip_x_arg + 87))
            print()
            
            quit()
            # mip_img = np.expand_dims(mip_x_img, axis=0)
            # mip_lbl = np.expand_dims(mip_y_img, axis=0)
            # print('mip_img',mip_img.shape)
            # print('mip_lbl',mip_lbl.shape)
            # # quit()
            # mip_img = sitk.GetImageFromArray(mip_img)
            # mip_lbl = sitk.GetImageFromArray(mip_lbl)
            
            # mip_img.SetDirection(direc)
            # mip_lbl.SetDirection(direc)
            # mip_img.SetOrigin(origin)
            # mip_lbl.SetOrigin(origin)
            # mip_img.SetSpacing(spac)
            # mip_lbl.SetSpacing(spac)
            
            # save_img_path = mip_img_16slice_path + data_name + '/'
            # save_lbl_path = mip_lbl_16slice_path + data_name + '/'
            # os.makedirs(save_img_path, exist_ok=True)
            # os.makedirs(save_lbl_path, exist_ok=True)
            
            # # print(save_img_path, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
            # # print(save_lbl_path)
            
            # sitk.WriteImage(mip_img, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
            # sitk.WriteImage(mip_lbl, os.path.join(save_lbl_path, ('%s'%(img_name[:-7]) + 'lbl_slice_all.nii.gz')))
            # quit()
        
            # mip_img_all[kk,:,:] = mip_x_img
            # mip_lbl_all[kk,:,:] = mip_y_img
            # np.save(os.path.join(mip_img_16slice_path, '%s'%(img_name[:-7]) + 'img_slice_%s.npy'%kk), mip_x_img)
            # np.save(os.path.join(mip_lbl_16slice_path, '%s'%(img_name[:-7]) + 'lbl_slice_%s.npy'%kk), mip_y_img)
            
        # quit()
    
def mip_generate_all_argdepth():
    img_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/imagesTs/'
    lbl_path_all = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/labelsTs/'
    mip_img_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_imagesTs/'
    mip_lbl_16slice_path = '/memory/shanwenqi/Vessel_seg/COSTA_Dataset/mip_labelsTs/'
    os.makedirs(mip_img_16slice_path, exist_ok=True)
    os.makedirs(mip_lbl_16slice_path, exist_ok=True)

    
    for img_name in os.listdir(img_path_all):
        img_path = img_path_all + img_name
        lbl_path = lbl_path_all + img_name
        
        img = sitk.ReadImage(img_path)# + img_name)
        lbl = sitk.ReadImage(lbl_path)# + img_name)


    # img_npy_path = mip_img_16slice_path + img_name[:-7]
    # lbl_npy_path = mip_lbl_16slice_path + img_name[:-7]
    
        spac = img.GetSpacing()
        origin = img.GetOrigin()
        direc = img.GetDirection()
    
        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(lbl)
            
        mip_x_img = np.max(x, 0)
        mip_y_img = np.max(y, 0)
        print(mip_x_img.shape)
        # quit()
        mip_img = np.expand_dims(mip_x_img, axis=0)
        mip_lbl = np.expand_dims(mip_y_img, axis=0)
        print('mip_img',mip_img.shape)
        print('mip_lbl',mip_lbl.shape)
        # quit()
        mip_img = sitk.GetImageFromArray(mip_img)
        mip_lbl = sitk.GetImageFromArray(mip_lbl)
        
        mip_img.SetDirection(direc)
        mip_lbl.SetDirection(direc)
        mip_img.SetOrigin(origin)
        mip_lbl.SetOrigin(origin)
        mip_img.SetSpacing(spac)
        mip_lbl.SetSpacing(spac)
        
        save_img_path = mip_img_16slice_path
        save_lbl_path = mip_lbl_16slice_path 
        os.makedirs(save_img_path, exist_ok=True)
        os.makedirs(save_lbl_path, exist_ok=True)
        
        # print(save_img_path, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
        # print(save_lbl_path)
        
        sitk.WriteImage(mip_img, os.path.join(save_img_path, ('%s'%(img_name[:-7]) + 'img_slice_all.nii.gz')))
        sitk.WriteImage(mip_lbl, os.path.join(save_lbl_path, ('%s'%(img_name[:-7]) + 'lbl_slice_all.nii.gz')))

# mip_generate_all_train_argdepth()
# mip_generate_all_test()

# mip_generate_all_train()
mip_generate_all_test()
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
    