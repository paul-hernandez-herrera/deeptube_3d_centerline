import centerline_tracing.construct_training_set as test
import centerline_tracing.input_output as io
import numpy as np

folder_imgs = r'C:\Users\jalip\Documentos\github\Tracing_centerline_single_brigth_structures\dataset\training\imgs'
folder_traces = r'C:\Users\jalip\Documentos\github\Tracing_centerline_single_brigth_structures\dataset\training\traces'

obj = test.construct_training_set(folder_imgs,folder_traces)
#swc = io.read_swc(r'C:\Users\jalip\Documentos\github\Tracing_centerline_single_brigth_structures\dataset\training\traces', 'calceina_2022_05_25_Exp3_stacks_TP0011.swc')
#cylinder = obj.construct_tubular_mask(np.array([20, 480, 640]), swc, 0)
#sphere = obj.construct_sphere(4)
#io.imwrite(r"C:\Users\jalip\Downloads\test_sphere.tif", sphere)
#print('done constructing cylinder')
#io.imwrite(r"C:\Users\jalip\Downloads\test_cylinder.tif", 255*cylinder)
#print (sphere[4,4,4])
#print(np.random.randint(100))