import pickle


def load_content_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model