import os

folder_path = r"C:\Users\lenna\Documents\Coding_Projects\multi_effect\overdrive_reverb"  # CHANGE THIS

for filename in os.listdir(folder_path):
    if not filename.endswith(".wav"):
        continue

    parts = filename.split("-")
    if len(parts) < 4:
        continue

    fx_code = parts[2]  # e.g., "1111"
    if len(fx_code) == 4 and fx_code[1:3] == "11":
        new_fx_code = fx_code[0] + "23" + fx_code[3]  # change "11" to "21"
        parts[2] = new_fx_code
        new_filename = "-".join(parts)

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")
