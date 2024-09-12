# %%
import pathlib
import gdown

# %%

FILES_DICT = {  
    # FASTPITCH
    "fastpitch_de.pth": {
        "path": "pretrained/fastpitch_de.pth",
        "url": "https://drive.google.com/file/d/1AqgObECvPp2SrYtklmNWuaDLmY1CU95Y/view?usp=sharing",
        "download": True,
    },
    # HIFIGAN
    "hifigan-thor.pth": {
        "path": "pretrained/hifigan-thor-v1/hifigan-thor.pth",
        "url": "https://drive.google.com/file/d/1HZJ2kRMysldjxr3MjAndyD-jA2mVucHT/view?usp=sharing",
        "download": True,
    },    
}

# %%

root_dir = pathlib.Path(__file__).parent

for file_dict in FILES_DICT.values():
    file_path = root_dir.joinpath(file_dict['path'])

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
    if file_path.exists():
        print(file_dict['path'], "already exists!")
    elif file_dict.get('download', True):
        print("Downloading ", file_dict['path'], "...")
        output_filepath = gdown.download(file_dict['url'], output=file_path.as_posix(), fuzzy=True)
