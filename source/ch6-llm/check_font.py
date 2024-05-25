import os
import subprocess
from collections import deque

# Define the font directory
font_dir = "/usr/share/fonts"

# Function to find symbolic links in the directory
def find_symlinks(directory):
    symlinks = []
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            path = os.path.join(root, name)
            if os.path.islink(path):
                symlinks.append(path)
    return symlinks

# Function to check for loops in symbolic links
def check_for_loops(symlink):
    visited = set()
    queue = deque([symlink])
    while queue:
        current = queue.popleft()
        if current in visited:
            return True
        visited.add(current)
        if os.path.islink(current):
            target = os.readlink(current)
            target_path = os.path.abspath(os.path.join(os.path.dirname(current), target))
            if os.path.exists(target_path):
                queue.append(target_path)
    return False

# Function to rename symlinks
def rename_symlinks(symlinks):
    renamed_symlinks = []
    for symlink in symlinks:
        if check_for_loops(symlink):
            print(f"Loop detected in {symlink}. Renaming temporarily...")
            new_name = symlink + ".bak"
            os.rename(symlink, new_name)
            renamed_symlinks.append((symlink, new_name))
    return renamed_symlinks

# Function to revert renamed symlinks
def revert_symlinks(renamed_symlinks):
    for original, renamed in renamed_symlinks:
        os.rename(renamed, original)

# Function to rebuild font cache
def rebuild_font_cache():
    try:
        subprocess.run(["fc-cache", "-r", "-v"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error rebuilding font cache: {e}")

# Main function
def main():
    print("Finding symbolic links in the font directory...")
    symlinks = find_symlinks(font_dir)
    if not symlinks:
        print("No symbolic links found.")
        return

    print(f"Found {len(symlinks)} symbolic links. Checking for loops and renaming them temporarily if loops are detected...")
    renamed_symlinks = rename_symlinks(symlinks)

    if not renamed_symlinks:
        print("No loops detected in symbolic links.")
        return

    print("Rebuilding font cache...")
    rebuild_font_cache()

    # Check if the issue is resolved
    issue_resolved = input("Is the looped directory issue resolved? (yes/no): ").strip().lower() == "yes"

    if issue_resolved:
        print("Issue resolved. Keeping the symbolic links renamed.")
    else:
        print("Issue not resolved. Reverting the symbolic links...")
        revert_symlinks(renamed_symlinks)
        print("Symbolic links reverted.")

if __name__ == "__main__":
    main()
