
from src.ml_logic import data, preprocessor, model
from src import params

def train():
    # 1. Load the data
    X, y = data.load_data()  # Adjust arguments as needed

    # 2. Preprocess the data
    X_preprocessed = preprocessor.preprocess(X)

    # 3. Build and train the model
    lip_model = model.build_model()
    lip_model.fit(X_preprocessed, y, epochs=10, batch_size=32)

    # 4. Save the model
    model.save_model(lip_model)

if __name__ == "__main__":
    train()
