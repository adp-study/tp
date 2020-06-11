import shutil
import os

r = 'E:/PythonProjects/HW/DL_projects/Dataset/Disc_2D_resampled_nonthickness_dilate_closing/Dataset'
p = os.listdir(r)

rl = [os.path.join(r, l) for l in p]
for ap in rl:
    pi = os.path.basename(ap)
    fl = [os.path.join(ap, f) for f in os.listdir(ap)]
    for fp in fl:
        fi = os.path.basename(fp)
        label_fp = fp.replace('closing/Dataset', 'closing/Label')
        new_fi = os.path.join('E:/PythonProjects/HW/DL_projects/Dataset/Disc_2D_resampled_nonthickness_dilate_closing_/Dataset', pi+'_'+fi)
        label_new_fi = os.path.join('E:/PythonProjects/HW/DL_projects/Dataset/Disc_2D_resampled_nonthickness_dilate_closing_/Label', pi+'_'+fi)

        if not os.path.exists(os.path.dirname(new_fi)):
            os.makedirs(os.path.dirname(new_fi))
        if not os.path.exists(os.path.dirname(label_new_fi)):
            os.makedirs(os.path.dirname(label_new_fi))
        print(new_fi, label_new_fi)
        shutil.copy(fp, new_fi)
        shutil.copy(label_fp, label_new_fi)

        # if '_' in os.path.basename(fp):
        #     print(fp)
        #     #os.remove(fp)