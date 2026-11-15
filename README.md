# Pokemon Predictor

    pip install -r requirements.txt
    python main.py                      # GUI (needs internet for PokeAPI)
    python main.py pikachu charizard    # CLI
    python evaluate.py                  # score all models on 50k real battles (offline)

Files: battle.py (simulator), data.py (PokeAPI), predictor.py (hybrid trained model),
evaluate.py (validation), visualize.py, main.py. validation_data/ = Kaggle
"Pokemon Challenge" battles + stats, and a static type chart.

Held-out results (2000 battles): raw simulator 69.5% acc / 1.93 log-loss;
hybrid model 94.7% acc / 0.155 log-loss. The labels come from a speed-dominated
dataset, so this measures fit to that data, not real-game accuracy.
