import shutil
import sys
from pathlib import Path

SRC_DIR = Path("../neuromodex_vnet_dbs")
WEIGHTS_DIR = Path("../neuromodex_vnet_dbs/weights")
BASE_EXTENSION_DIR = Path(".")

PLUGINS = {
    "BrainSegmentation": {
        "files": [
            "SegmentationPipeline.py",
            "models/SegmentationModelBase.py",
            "models/CNNBasedSegmentationModel.py",
            "models/GMM.py",
            "models/ProbabilisticBasedSegmentationModel.py",
            "models/architecture/VNet.py",
            "core/BaseProcessor.py",
            "data/postprocessing.py",
            "data/preprocessing.py",
            "data/sitk_transform.py",
        ],
        "copy_weights": True,
    },
    "ConductivityMapping": {
        "files": [
            "ConductivityProcessingPipeline.py",
            "core/BaseProcessor.py",
            "models/ConductivityMapper.py",
            "data/sitk_transform.py",
        ],
        "copy_weights": False,
    }
}

def prompt_plugin_selection():
    print("🛠️  Plugin Packaging Tool\n")
    print("📦 Available Plugins:")
    for i, name in enumerate(PLUGINS, 1):
        print(f"  {i}) {name}")
    print("  0) All\n")

    selection = input("🔧 Package which plugin(s)? Enter numbers separated by comma (e.g. 1,2 or 0): ").strip()
    if not selection:
        print("❌ No selection made.")
        sys.exit(1)

    selected = []
    if "0" in selection:
        return list(PLUGINS.keys())

    for num in selection.split(","):
        num = num.strip()
        try:
            idx = int(num) - 1
            plugin = list(PLUGINS.keys())[idx]
            selected.append(plugin)
        except (IndexError, ValueError):
            print(f"❌ Invalid selection: {num}")
            sys.exit(1)

    return selected

def package_plugin(plugin_name):
    config = PLUGINS[plugin_name]
    source_path = BASE_EXTENSION_DIR / plugin_name
    source_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Packaging: {plugin_name}")
    for relative_path in config["files"]:
        src = SRC_DIR / relative_path
        dst = source_path / SRC_DIR.name / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"   ✔ {relative_path}")

    if config["copy_weights"]:
        print("📥 Copying weights...")
        shutil.copytree(WEIGHTS_DIR, source_path / WEIGHTS_DIR.parent.name / "weights", dirs_exist_ok=True)

def main():
    selected_plugins = prompt_plugin_selection()

    for plugin in selected_plugins:
        package_plugin(plugin)

    print("\n✅ Packaging complete. Run the installer next.")
    input()

if __name__ == "__main__":
    main()
