from typing import Dict, Optional, Tuple
from pathlib import Path
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense,  RNN
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa
from flwr.common import Parameters, NDArrays, ndarrays_to_parameters, MetricsAggregationFn, FitRes, parameters_to_ndarrays
from tensorflow.keras.optimizers import Adam
from flwr.common import Metrics
from typing import List, Tuple
import flwr as fl
import pandas as pd
import numpy as np

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=16, input_length=10))
    model.add(RNN(tfa.rnn.LayerNormLSTMCell(64, recurrent_dropout = 0.6), return_sequences=False))
    #model.add(LSTM(64))
    model.add(Dense(10000, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    
    
    # Create strategy
    strategy = fl.server.strategy.FedProx(
        fraction_fit=1,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        fit_metrics_aggregation_fn = weighted_average,
        mu=0.01,
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    history = fl.server.start_server(
        server_address= "[::]:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )
    

    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies) / sum(examples))}
    
def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 
    return {"val_steps": val_steps}




if __name__ == "__main__":
    main()
