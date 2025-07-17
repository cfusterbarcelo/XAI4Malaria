# remove_xai_leak.py

import os

def main():
    folder = r"D:/Data/Malaria/MalariaSingleCellDataset/cell_images/Parasitized"
    removed = 0

    # patterns to match at the end of the filename
    bad_suffixes = ("_0_0.png", "_0_1.png", "_1_0.png", "_1_1.png")

    for fname in os.listdir(folder):
        lname = fname.lower()
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue

        # remove if it contains true/pred or matches one of the bad suffixes
        if "true" in lname or "pred" in lname or any(lname.endswith(suffix) for suffix in bad_suffixes):
            print(f"Removing: {path}")
            os.remove(path)
            removed += 1

    print(f"\nDone. Removed {removed} files.")

if __name__ == "__main__":
    main()
