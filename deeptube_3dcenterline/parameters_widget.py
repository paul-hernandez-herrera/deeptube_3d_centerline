import ipywidgets as widgets
from IPython.display import display

def set_parameters_generate_training_set(obj):
    print('------------------------------')
    print('\033[47m' '\033[1m' 'REQUIRED PARAMETERS' '\033[0m')
    print('------------------------------')

    folder_imgs_path_w = set_parameter_text('Folder images path:', 'Insert path here')   
    folder_swc_path_w = set_parameter_text('Folder swc files path:', 'Insert path here')

    print('------------------------------')
    print('\033[47m' '\033[1m' 'OPTIONAL PARAMETERS' '\033[0m')
    print('------------------------------')
    
    folder_output_default = 'Default: "Folder images path"/training_set/'
    folder_output_path_w = set_parameter_text('Folder output path:', folder_output_default)
    patch_size_img_w = set_parameter_Int('Patch size (length x length): ', obj.patch_size_img)
    number_patches_w = set_parameter_Int('# sub-images per image: ', obj.number_patches)
    radius_tubular_mask_w = set_parameter_Int('Radius tubular mask (GT): ', obj.radius_tubular_mask)
    draw_head_w = set_parameter_checkbox('Draw head', obj.draw_head)
    percentile_w = set_parameter_intSlider('Percentile normalization', obj.norm_perc_low, obj.norm_perc_high)



    parameters = {'folder_imgs_path_w' : folder_imgs_path_w,
                  'folder_swc_path_w' : folder_swc_path_w,
                  'patch_size_img_w': patch_size_img_w,
                  'number_patches_w': number_patches_w,
                  'radius_tubular_mask_w': radius_tubular_mask_w,
                  'percentile_w': percentile_w,
                  'draw_head_w': draw_head_w,
                  'folder_output_path_w': folder_output_path_w,
                  'folder_output_default': folder_output_default}
    
    return parameters

def parameters_UNET_training():
    print('------------------------------')
    print('\033[47m' '\033[1m' 'REQUIRED PARAMETERS' '\033[0m')
    print('------------------------------')

    folder_input_w = set_parameter_text('Folder path input images:', 'Insert path here')   
    folder_target_w = set_parameter_text('Folder path target mask:', 'Insert path here')


    parameters = {'folder_input_w' : folder_input_w,
                  'folder_target_w' : folder_target_w
                  }
    
    return parameters

def set_parameter_intSlider(string_name, low_val, high_val):
    widget_intSlider = widgets.IntRangeSlider(
        value=[low_val, high_val],
        min=0,
        max=100,
        step=1,
        description=string_name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(flex='1 1 auto', width='auto')
    )
    display(widget_intSlider)
    return widget_intSlider

def set_parameter_text(string_name, string_default_val):
    widget_text = widgets.Text(
        value = '',
        placeholder = string_default_val,
        description= '',
        disable = False,
        layout=widgets.Layout(flex='1 1 auto', width='auto'))
    
    a = widgets.HBox([widgets.Label(string_name, layout=widgets.Layout( width='200px')),widget_text])
    display(a)
    return widget_text 

def set_parameter_Int(string_name, default_value):
    widget_int =widgets.IntText(
        value = default_value,
        description = '',
        disabled=False,
        layout=widgets.Layout(flex='1 1 auto', width='auto'))    
    
    a = widgets.HBox([widgets.Label(string_name, layout=widgets.Layout( width='200px')), widget_int])
    display(a)
    return widget_int 

def set_parameter_checkbox(string_name, default_value):
    widget_checkbox = widgets.Checkbox(
        value=default_value,
        description= string_name,
        disabled=False,
        indent=False,
    )
    display(widget_checkbox)
    return widget_checkbox     