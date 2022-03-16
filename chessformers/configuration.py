import yaml


def get_configuration(yaml_path: str):
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        return yaml_dict