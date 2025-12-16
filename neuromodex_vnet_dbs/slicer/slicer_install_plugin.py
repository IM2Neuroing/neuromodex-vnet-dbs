import os
import sys
import platform
import shutil
import configparser
from pathlib import Path

SLICER_VERSION = "5.8.0"
BASE_EXTENSION_DIR = Path(__file__).parent

def prompt_plugin_selection(available_plugins):
    print("\U0001F9E0 Slicer Plugin Installer")
    print(f"\U0001F4E6 Compatible with Slicer ≥ {SLICER_VERSION}\n")
    print("\U0001F4CC Available Plugins:")
    for i, name in enumerate(available_plugins, 1):
        print(f"  {i}) {name}")
    print("  0) All\n")

    selection = input("❓ Install which plugin(s)? Enter numbers separated by comma (e.g., 1,2 or 0): ").strip()
    if not selection:
        print("❌ No selection made.")
        sys.exit(1)

    if "0" in selection:
        return available_plugins

    selected = []
    for num in selection.split(","):
        num = num.strip()
        try:
            idx = int(num) - 1
            plugin = available_plugins[idx]
            selected.append(plugin)
        except (IndexError, ValueError):
            print(f"❌ Invalid selection: {num}")
            sys.exit(1)

    return selected

def find_slicer_installation():
    print("\U0001F50D Searching for Slicer installation...")
    system = platform.system()

    if system == "Windows":
        standard_dir = Path(os.environ['LOCALAPPDATA']) / "slicer.org"
        for folder in standard_dir.iterdir():
            if folder.is_dir() and "slicer" in folder.name.lower():
                parts = folder.name.split(" ")
                if len(parts) > 1 and parts[1] >= SLICER_VERSION:
                    slicer_exe = folder / "Slicer.exe"
                    if slicer_exe.exists():
                        print(f"✅ Found Slicer: {folder}")
                        return find_ini_and_extension_folder(folder)

        print("⚠️ Could not auto-detect Slicer.")
        user_path = Path(input("\U0001F4C1 Manually enter main Slicer path: "))
        if not user_path.exists():
            print("❌ Invalid path.")
            sys.exit(1)
        return find_ini_and_extension_folder(user_path)

    elif system == "Darwin":
        return {
            "ini_file": Path.home() / "Library/Application Support/NA-MIC" / f"Slicer-{SLICER_VERSION}.ini",
            "extensions_dir": Path.home() / "Library/Application Support/Slicer" / f"Extensions-{SLICER_VERSION[:3]}",
            "slicer_root": Path.home() / "Library/Application Support/Slicer"
        }

    elif system == "Linux":
        return {
            "ini_file": Path.home() / ".config/NA-MIC" / f"Slicer-{SLICER_VERSION}.ini",
            "extensions_dir": Path.home() / ".config/Slicer" / f"Extensions-{SLICER_VERSION[:3]}",
            "slicer_root": Path.home() / ".config/Slicer"
        }

    else:
        print("❌ Unsupported system.")
        sys.exit(1)

def find_ini_and_extension_folder(slicer_root_dir):
    ini_dir = slicer_root_dir / "slicer.org"
    ini_file = next((f for f in ini_dir.glob("*.ini")), None)
    extension_folder = next((d for d in ini_dir.iterdir() if d.is_dir()), None)

    if not ini_file or not extension_folder:
        print("❌ Could not find .ini or extension folder.")
        sys.exit(1)

    return {
        "ini_file": ini_file,
        "extensions_dir": extension_folder,
        "slicer_root": slicer_root_dir
    }

def install_plugin(plugin_name, extension_folder, ini_file, slicer_root):
    src_path = BASE_EXTENSION_DIR / plugin_name
    dst_path = extension_folder / plugin_name
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    print(f"📁 Copied {plugin_name} to Slicer extension folder.")

    config = configparser.ConfigParser()
    config.read(ini_file, encoding="utf-8")

    if "Modules" not in config:
        config["Modules"] = {}

    current_paths = config["Modules"].get("AdditionalPaths", "")
    if current_paths == "@Invalid()":
        current_paths = ""

    paths = [p.strip() for p in current_paths.replace(";", ",").split(",") if p.strip()]
    relative_path = Path(str(os.path.relpath(dst_path, slicer_root))).as_posix()

    if relative_path not in paths:
        paths.append(relative_path)
        config["Modules"]["AdditionalPaths"] = ", ".join(paths)
        with open(ini_file, "w", encoding="utf-8") as f:
            config.write(f)
        print(f"✅ INI updated with: {relative_path}")
    else:
        print(f"ℹ️ INI already contains: {relative_path}")

def interactive_installation():
    available_plugins = [d.name for d in BASE_EXTENSION_DIR.iterdir() if d.is_dir()]
    selected_plugins = prompt_plugin_selection(available_plugins)

    slicer_info = find_slicer_installation()
    ini_file = slicer_info["ini_file"]
    extension_folder = slicer_info["extensions_dir"]
    slicer_root = slicer_info["slicer_root"]

    for plugin in selected_plugins:
        print(f"\n🚀 Installing: {plugin}")
        install_plugin(plugin, extension_folder, ini_file, slicer_root)

    print("\n🎉 Installation complete!")
    print("🔁 Restart Slicer to activate the plugins.")
    if __name__ == "__main__":
        input()

if __name__ == "__main__":
    interactive_installation()
