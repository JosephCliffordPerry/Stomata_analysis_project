library(reticulate)

Sys.setenv(RETICULATE_PYTHON = "managed")
py_require(packages = c("nd2", "tifffile"), python_version = "3.12.4")

convert_code <- "
import os
import nd2
import tifffile
import numpy as np

def convert_all_nd2_in_folder(input_dir, output_dir='tif_output'):
    os.makedirs(output_dir, exist_ok=True)

    nd2_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.nd2')]
    if not nd2_files:
        print('⚠️ No ND2 files found in', input_dir)
        return

    for nd2_file in nd2_files:
        input_path = os.path.join(input_dir, nd2_file)
        sample_name = os.path.splitext(nd2_file)[0]

        print(f'🔄 Converting {nd2_file} -> {output_dir}')

        with nd2.ND2File(input_path) as f:
            data = f.asarray()
            ndim = data.ndim

            if ndim == 2:
                out_name = f'{sample_name}_frame_0000.tif'
                out_path = os.path.join(output_dir, out_name)
                tifffile.imwrite(out_path, data)
            else:
                for i in range(data.shape[0]):
                    out_name = f'{sample_name}_frame_{i:04d}.tif'
                    out_path = os.path.join(output_dir, out_name)
                    tifffile.imwrite(out_path, data[i])

        print(f'✅ Finished {nd2_file}')
"

py_run_string(convert_code)

# Run it — all frames go into the same folder
py$convert_all_nd2_in_folder("D:/stomata/training_nd2", "D:/stomata/training_tifs")
