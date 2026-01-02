import os
import glob

def clear_bird_labels():
    dataset_path = r"e:\Computer_Vision\vision-based-uav-detection\Dataset"
    splits = ["train", "valid", "test"]
    
    total_cleared = 0
    
    for split in splits:
        labels_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(labels_dir):
            print(f"Directory not found: {labels_dir}")
            continue
            
        # pattern for Bird files: BTR*.txt, BT*.txt, BV*.txt -> starts with B
        files = glob.glob(os.path.join(labels_dir, "B*.txt"))
        
        print(f"Processing {split}: Found {len(files)} Bird label files.")
        
        for file_path in files:
            # check if file is not empty before clearing (optional, but good for stats)
            if os.path.getsize(file_path) > 0:
                with open(file_path, 'w') as f:
                    f.write("") # Clear content
                total_cleared += 1
                
    print(f"Total Bird label files cleared: {total_cleared}")

if __name__ == "__main__":
    clear_bird_labels()
