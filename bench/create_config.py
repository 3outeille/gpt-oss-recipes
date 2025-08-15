import argparse
import yaml
from pathlib import Path

def create_config():
    parser = argparse.ArgumentParser(description="Create YAML configuration files for benchmarking.")
    parser.add_argument("--zero3", action="store_true", help="Flag for Zero3.")
    parser.add_argument("--tp", action="store_true", help="Flag for EP.")
    parser.add_argument("--peft", action="store_true", help="Flag for PEFT.")
    parser.add_argument("--megablocks", action="store_true", help="Flag for Megablocks.")
    parser.add_argument("--flash", action="store_true", help="Flag for Flash Attention.")
    args = parser.parse_args()

    script_path = Path(__file__).parent
    template_path = script_path / "template.yaml"
    config_dir = script_path / "configs"
    config_dir.mkdir(exist_ok=True)

    with open(template_path, 'r') as f:
        config_data = yaml.safe_load(f)

    name_parts = []
    
    if args.zero3:
        name_parts.append("zero3")
        config_data["use_tp"] = False
    elif args.tp:
        name_parts.append("tp")
        config_data["use_tp"] = True

    if args.peft:
        name_parts.append("lora")
        config_data["use_peft"] = True

    if args.megablocks:
        name_parts.append("megablocks")
        config_data["use_kernels"] = True

    if args.flash:
        name_parts.append("flash")
        config_data["attn_implementation"] = "kernels-community/vllm-flash-attn3"
    else:
        config_data["attn_implementation"] = "eager"

    # Update output directory
    output_dir_name = "-".join([part for part in name_parts])
    
    config_data["output_dir"] = f"gpt-20b-oss-{output_dir_name}"
    config_data["run_name"] = config_data["output_dir"]

    # Generate config file
    config_filename = "_".join(name_parts) + ".yaml"
    config_filepath = config_dir / config_filename

    with open(config_filepath, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

    print(f"Successfully generated config: {config_filepath}")

if __name__ == "__main__":
    create_config()
