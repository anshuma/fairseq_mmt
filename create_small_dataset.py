import os

# Function to read the list of .jpg files from the .txt file
def read_jpg_list(txt_file):
    with open(txt_file, 'r') as file:
        jpg_list = file.read().splitlines()
    return set(jpg_list)

# Function to delete .jpg files not in the list
def delete_unlisted_jpgs(folder, jpg_list):
    for file in os.listdir(folder):
        if file.endswith('.jpg') and file not in jpg_list:
            os.remove(os.path.join(folder, file))
            print(f"Deleted: {file}")

# Main function
def main():
    txt_file = 'list_of_files.txt'  # Path to the .txt file
    folder = '/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/flickr30k-images'  # Path to the folder containing .jpg files

    # Read the list of .jpg files from the .txt file
    jpg_list_1 = read_jpg_list('/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/train.txt')
    jpg_list_2 = read_jpg_list('/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/val.txt')
    jpg_list_3 = read_jpg_list('/Users/anshumashuk/git/MTech_IITHyd/IITHyd_Capstone/final_Capstone_experiments/fairseq_mmt/small_dataset/flickr30k/test_2016_flickr.txt')
    final_jpg_list = jpg_list_1.union(jpg_list_2).union(jpg_list_3)
    # Delete unlisted .jpg files
    delete_unlisted_jpgs(folder, final_jpg_list)

if __name__ == '__main__':
    main()
