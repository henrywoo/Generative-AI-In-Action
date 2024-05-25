from datasets import load_dataset
import ast

# Load the HumanEval dataset
human_eval = load_dataset("openai_humaneval")

# Function to run the test cases
def run_test_cases(func, test_cases):
    for test_case in test_cases:
        input_data = ast.literal_eval(test_case['input'])
        expected_output = test_case['output']
        # Evaluate the function and compare with the expected output
        result = func(*input_data) if isinstance(input_data, tuple) else func(input_data)
        assert result == expected_output, f"Test failed: {input_data} => {result}, expected: {expected_output}"
    print("All test cases passed.")

# Display and execute test cases for 3 rows from the HumanEval dataset
for i in range(3):
    row = human_eval['test'][i]
    prompt = row['prompt']
    code = row['canonical_solution']
    test_cases = row['test']

    print(f"Prompt {i+1}: {prompt}")
    print(f"Code {i+1}:\n{code}")

    # Write the function code to a string with proper indentation
    exec_globals = {}
    exec_locals = {}

    # Add an additional line to ensure the code block executes properly
    code_block = f"""{prompt}
{code}
"""

    print(f"\n----\n{code_block}\n----\n")
    exec(code_block, exec_globals, exec_locals)
    func_name = list(exec_locals.keys())[0]
    func = exec_locals[func_name]

    # Run the test cases for the function
    run_test_cases(func, test_cases)
    print("\n")
