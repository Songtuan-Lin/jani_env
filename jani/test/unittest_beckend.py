import sys
sys.path.append("/home/garrick/codes/jani_env/jani/engine/build")

from backend import JANIEngine

class JANIDemo:
    def __init__(self, jani_model_path, jani_property_path, start_states_path, objective_path, failure_property_path, seed):
        self.engine = JANIEngine(jani_model_path, jani_property_path, start_states_path, objective_path, failure_property_path, seed)

    def query_guards_for_action(self, action_id):
        all_guards = self.engine.test_guards_for_action(action_id)
        for guard_expression in all_guards:
            answer = input("Continue? (y/n): ").strip().lower()

            while answer not in ("y", "n"):
                answer = input("Please type y or n: ").strip().lower()
            
            if answer == "n":
                break

            print(f"Guard Expression: {guard_expression}")
    
    def query_destinations_for_action(self, action_id):
        all_destinations = self.engine.test_destinations_for_action(action_id)
        for destination_set in all_destinations:
            answer = input("Continue? (y/n): ").strip().lower()

            while answer not in ("y", "n"):
                answer = input("Please type y or n: ").strip().lower()
            
            if answer == "n":
                break

            print("Destination Assignments:")
            for destination in destination_set:
                print("Destination Assignments:")
                for var_name, expr in destination.items():
                    print(f"  {var_name}: {expr}")
            print("-----")

if __name__ == "__main__":
    jani_model_path = sys.argv[1]
    jani_property_path = sys.argv[2]
    start_states_path = sys.argv[3]
    objective_path = sys.argv[4]
    failure_property_path = sys.argv[5]
    seed = int(sys.argv[6])
    demo = JANIDemo(jani_model_path, jani_property_path, start_states_path, objective_path, failure_property_path, seed)

    input_action = input("Enter action ID to query guards: ").strip()
    while input_action.lower() != "exit":
        try:
            action_id = int(input_action)
            demo.query_guards_for_action(action_id)
            demo.query_destinations_for_action(action_id)
        except ValueError:
            print("Invalid action ID. Please enter an integer.")

        input_action = input("Enter action ID to query guards (or type 'exit' to quit): ").strip()
                
    
