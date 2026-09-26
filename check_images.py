import os
import hashlib

folders = [
    "fist",
    "palm",
    "peace",
    "thumbs_up"
]

print("\n==============================")
print("DATASET IMAGE CHECK")
print("==============================")

all_hashes = {}

for folder in folders:

    path = os.path.join("dataset", folder)

    files = [
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n{folder.upper()}")
    print("Images:", len(files))

    hashes = []

    for filename in files:

        filepath = os.path.join(path, filename)

        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        hashes.append(file_hash)

    all_hashes[folder] = hashes

    print("Unique images:", len(set(hashes)))

print("\n==============================")
print("DUPLICATE CHECK")
print("==============================")

for i in range(len(folders)):
    for j in range(i + 1, len(folders)):

        folder1 = folders[i]
        folder2 = folders[j]

        common = set(all_hashes[folder1]) & set(all_hashes[folder2])

        print(
            f"{folder1} vs {folder2}: "
            f"{len(common)} identical files"
        )

print("==============================")