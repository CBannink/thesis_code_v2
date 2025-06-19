import pickle
import numpy as np
import argparse
import zmq
import json
import anndata


class CellOracleService:
    """A service that loads a CellOracle model and performs inference."""

    def __init__(self, model_path, adata_path):
        """
        Initialize the service with a CellOracle model and AnnData object.

        Args:
            model_path: Path to the saved CellOracle model
            adata_path: Path to the AnnData object
        """
        print("Loading CellOracle model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        print("Service initialized and ready for inference.")

    def simulate_perturbation(self, cell_index, gene_name, perturbation_value):
        """
        Simulate a perturbation using CellOracle.

        Args:
            cell_index: Index of the cell to perturb
            gene_name: Name of the gene to perturb
            perturbation_value: Value of the perturbation

        Returns:
            new_cell_index: Index of the resulting cell state
        """
        # Create perturbation dictionary for CellOracle
        perturbation = {gene_name: perturbation_value}

        # Run CellOracle simulation
        simulation_result = self.model.simulate_perturbation(
            cell_indices=[cell_index],
            perturbation_dict=perturbation,
            n_steps=1
        )

        # Get the resulting cell state
        new_cell_index = simulation_result.get_resulting_cell_index(cell_index)

        return new_cell_index


def start_service(model_path, adata_path, port=5555):
    """
    Start the CellOracle service and listen for requests.

    Args:
        model_path: Path to the saved CellOracle model
        adata_path: Path to the AnnData object
        port: Port to listen on
    """
    # Initialize the service
    service = CellOracleService(model_path, adata_path)

    # Set up ZMQ socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    print(f"CellOracle service listening on port {port}...")

    while True:
        # Wait for request
        message = socket.recv_json()

        # Process request
        try:
            cell_index = message["cell_index"]
            gene_name = message["gene_name"]
            perturbation_value = message["perturbation_value"]

            # Perform simulation
            new_cell_index = service.simulate_perturbation(
                cell_index, gene_name, perturbation_value
            )

            # Send response
            response = {"success": True, "new_cell_index": int(new_cell_index)}

        except Exception as e:
            response = {"success": False, "error": str(e)}

        socket.send_json(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CellOracle Inference Service")
    parser.add_argument("--model", required=True, help="Path to the CellOracle model")
    parser.add_argument("--adata", required=True, help="Path to the AnnData object")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")

    args = parser.parse_args()

    start_service(args.model, args.adata, args.port)
