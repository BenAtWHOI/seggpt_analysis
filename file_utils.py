import os
import re


################################################################################
## Map the file prefixes to the correct subfolders for AMPLIfy yolo data
################################################################################
def get_subfolder(filename):
    dirs = {
        'Basler_avA2300-25gm__22955661__20201013_202814532': 'EN657_13Oct2020_996',
        'Basler_avA2300-25gm__22955661__20201013_205721189': 'EN657_13Oct2020_997',
        'Basler_avA2300-25gm__22955661__20201013_212541536': 'EN657_13Oct2020_998',
        'Basler_avA2300-25gm__22955661__20201013_220305958': 'EN657_13Oct2020_999',
        'Basler_avA2300-25gm__22955661__20201013_22030598': 'EN657_13Oct2020_007',
        'Basler_avA2300-25gm__22955661__20201014_145015823': 'EN657_14Oct2020_001',
        'Basler_avA2300-25gm__22955661__20201014_152334043': 'EN657_14Oct2020_002'
    }
    patterns = {re.compile(f"^{re.escape(key)}\\d*"): value for key, value in dirs.items()}
    for pattern, subfolder in patterns.items():
        if pattern.match(filename):
            return subfolder
    return 'tmp'


################################################################################
## Replace the given pattern for files in the directory
################################################################################
def normalize_filename(directory, pattern, replacement):
    for filename in os.listdir(directory):
        if pattern in filename:
            new_filename = filename.replace(pattern, replacement)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)


################################################################################
## Clear all files in the directory
################################################################################
def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        else:
            clear_directory(file_path)