def postprocess_predictions(x):
    return (x >= 0.5).astype(int)