from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "qwen/CodeQwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/CodeQwen1.5-7B-Chat")

prompt = """Complete the following code so that it can be compiled successfully. Show me the complete code!

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class IncompleteCode {
    private static Map<String, String> users = new HashMap<>();

    // Initialize some users
    static {
        users.put("user1", "password1");
        users.put("user2", "password2");
        // Add more users if needed
    }

    public static boolean authenticateUser(String username, String password) {
        // Check if the username exists and the password matches
        return users.containsKey(username) && users.get(username).equals(password);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Welcome to User Authentication System!");

        while (true) {
            System.out.print("Enter username: ");
            String username = scanner.nextLine();

            System.out.print("Enter password: ");
            String password = scanner.nextLine(
"""
messages = [
    {"role": "system", "content": "You are a helpful assistant who can complete java code so that it can be compiled successfully."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)