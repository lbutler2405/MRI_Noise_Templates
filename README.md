# MRI_Noise_Templates
Here, we provide the code with multiple functionalities to generate MRI noise templates, with the aim of providing the research community with flexibility for future MRI research. 

### Example code to: 
## 1) generate a block of rician noise as a pickle file (other formats are available)
## 2) Convert the templates to nifti format
## 3) Mask out the nifti template with the mask of the participant

--- --- 

# Command that generates a block of rician noise as a pickle file
    noise_generation_command="python3 $noise_py
                            --action generate
                            --nifti_file $raw_func
                            --output_dir $iter_folder
                            --template_shape 90 104 72 284
                            --num_templates 1
                            --output_format pickle
                            --template_base_name "template_$iteration"
                            --random_seed $iteration
                            --use_mask compute
                            --verbose"

    # Run the command
    #$noise_generation_command

  # Convert the templates to nifti format
    noise_conversion_command="python3 $noise_py
                            --action convert
                            --data_dir $iter_folder
                            --nifti_file $raw_func                                  
                            --output_dir $iter_folder
                            --which_mode random
                            --random_count 1      
                            --file_extension .pkl
                            --verbose"

    # Run the command
    #$noise_conversion_command

  # Delete the pickle template to save space
    #rm "$iter_folder/template_${iteration}_1.pkl"

  # Mask out the nifti template with the mask of the participant
    noise_fill_command="python $noise_py
                        --action fill
                        --data_dir $iter_folder  
                        --output_dir $iter_folder
                        --nifti_file $raw_func
                        --user_mask $sub_mask
                        --file_extension .nii.gz
                        --which_mode random
                        --random_count 1
                        --output_formats nifti
                        --random_seed $iteration
                        --verbose"  
   
    # Run the command
    #$noise_fill_command
