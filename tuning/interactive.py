import subprocess
import json
import os


def get_user_input():
    print("Let's choose a trait, parameter, and possible values")


    # Parameter selection
    print("Select a parameter:")
    parameters = ['Learning Rate', 'Batch Size', 'Number of Heads', 'Model Dimension',
                  'Number of Layers', 'Dropout Rate', 'MLP Dimension']
    for i, param in enumerate(parameters, 1):
        print(f"{i}. {param}")
    param_choice = int(input("Enter the number of the parameter you want to select: "))
    parameter = parameters[param_choice - 1]

    # Parameter values input and validation
    while True:
        param_values = input('Enter parameter values to test (comma or space-separated): ').strip()
        if not param_values:
            print("Please enter valid parameter values.")
            continue

        if parameter == 'Learning Rate' or parameter == 'Dropout Rate':
            try:
                # Split input by either commas or spaces, strip each item, and convert to float
                param_values_list = [float(val.strip()) for val in param_values.replace(',', ' ').split()]
                break  # Exit the loop if the conversion is successful
            except ValueError:
                print("Invalid input! Please enter a list of numbers (floats), separated by commas or spaces.")

        else:
            try:
                # Split input by either commas or spaces, strip each item, and convert to float
                param_values_list = [int(val.strip()) for val in param_values.replace(',', ' ').split()]
                break  # Exit the loop if the conversion is successful
            except ValueError:
                print("Invalid input! Please enter a list of numbers (floats), separated by commas or spaces.")
    

    
    # Trait selection
    print("Select a trait:")
    traits = ['Yield', '100 seed Weight', 'Days to flowering', 'Days to Maturity', 'Moisture']
    for i, trait in enumerate(traits, 1):
        print(f"{i}. {trait}")
    trait_choice = int(input("Enter the number of the trait you want to select: "))
    trait = traits[trait_choice - 1]

    return parameter, param_values_list, trait

# Call the function and assign the returned values
parameter, param_values, trait = get_user_input()

# Set environment variables
os.environ["TRAIT"] = trait
os.environ["PARAMETER"] = parameter
os.environ["PARAM_VALUES"] = ",".join(map(str, param_values))  # Convert list to comma-separated string

# Save parameters to the file
with open("chosenParams.txt", "w") as file:
    params = {
        'trait': trait,
        'parameter': parameter,
        'param_values': param_values
    }
    json.dump(params, file, indent=4)

    
def submit_job_to_cluster(script_path):
    try:
        # Run the sbatch command
        result = subprocess.run(
            ['sbatch', script_path],
            check=True, text=True, capture_output=True
        )
        
        # Capture the output and check if the submission was successful
        print("Job submitted successfully!")
        print(f"Job ID: {result.stdout.strip()}") 
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"stderr: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}") 

script_path = "optimizeTF.sh"
submit_job_to_cluster(script_path)


