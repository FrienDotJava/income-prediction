import yaml
def load_params(param_path: str) -> dict:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        raise Exception(f"Error loading parameters: {e}")