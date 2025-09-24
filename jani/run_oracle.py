from .oracle import TarjanOracle
from .core import *


def main():
    import argparse
    import pandas as pd

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

    parser = argparse.ArgumentParser(description="run oracle.")
    parser.add_argument("model_file", type=str, help="Path to the JANI model file.")
    parser.add_argument("property_file", type=str, help="Path to the output JSON file.")
    parser.add_argument("state_file", type=str, help="Path to the file containing states.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    
    args = parser.parse_args()

    model = JANI(model_file=args.model_file, property_file=args.property_file)

    oracle = TarjanOracle(model)
    df = pd.read_csv(args.state_file, header=None)
    num_incorrect = 0
    processed_states = 0
    with Progress(
            SpinnerColumn(),
            TextColumn("Processed: {task.fields[processed_states]}"),
            TextColumn("•"),
            TextColumn("Incorrect: {task.fields[num_incorrect]}"),
            TextColumn("•"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
        task = progress.add_task("Processing states", total=df.shape[0], num_incorrect=num_incorrect, processed_states=processed_states)

        for row in range(df.shape[0]):
            state_vector = df.iloc[row, :-5].to_numpy()
            state = State.from_vector(state_vector, model.get_constants_variables())
            processed_states += 1
            if df.iloc[row, -5] == -1.0:
                # terminal state, skip
                progress.update(task, processed_states=processed_states)
                progress.advance(task, advance=1)
                continue
            safe = oracle.is_safe(state)
            if int(safe) != df.iloc[row, -1]:
                num_incorrect += 1
                df.iloc[row, -1] = int(safe)
                progress.update(task, num_incorrect=num_incorrect, processed_states=processed_states)
            else:
                progress.update(task, processed_states=processed_states)
            progress.advance(task, advance=1)

        print(f"Number of incorrect states: {num_incorrect}")

    df.to_csv(args.output_file, header=False, index=False)

if __name__ == "__main__":
    main()