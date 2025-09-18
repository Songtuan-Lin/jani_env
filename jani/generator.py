import json

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from .core import JANI


class StateGenerator:
    def __init__(self, model: JANI, num_states: int) -> None:
        self._model = model
        self._num_states = num_states

    def generate_states(self, output_file: str) -> None:
        def create_json(states: set) -> dict:
            json_obj = {
                "op": "states-values",
                "values": list()
            }
            for state in states:
                assignment = {"variables": list()}
                for var_name, var in state.variable_dict.items():
                    assignment["variables"].append({
                        "var": var_name,
                        "value": var.value
                    })
                json_obj["values"].append(assignment)
            return json_obj

        states = set()
        
        # Create a progress bar that matches pip's style
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("Generating states", total=self._num_states)
            
            for _ in range(self._num_states):
                state = self._model.reset()
                states.add(state)
                progress.advance(task)

        print(f"Generated {len(states)} unique states out of {self._num_states} attempts.")

        json_data = create_json(states)
        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate random states from a JANI model.")
    parser.add_argument("model_file", type=str, help="Path to the JANI model file.")
    parser.add_argument("property_file", type=str, help="Path to the property file.")
    parser.add_argument("num_states", type=int, help="Number of states to generate.")
    parser.add_argument("output", type=str, help="Output file to save the generated states.")
    args = parser.parse_args()

    model = JANI(args.model_file, property_file=args.property_file, block_previous=True, block_all=True)
    generator = StateGenerator(model, args.num_states)
    generator.generate_states(args.output)

if __name__ == "__main__":
    main()