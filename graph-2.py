import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import io

# Function to generate a graph based on user-provided formula or predefined equations
def generate_graph(formula, x_range):
    plt.figure(figsize=(10, 6))
    
    try:
        x = np.linspace(x_range[0], x_range[1], 400)
        y = eval(formula)
        plt.plot(x, y, label=formula)
    except Exception as e:
        print(f"Error in generating graph: {e}")
        return None
    
    plt.title('Graph of the function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.show()
    
    return buf

# Function to generate a description of the graph
def describe_graph(formula):
    description = f"The graph represents the function {formula}. "
    # Add some unique aspects to the description
    if 'sin' in formula:
        description += "It is a sinusoidal wave, showing periodic oscillations."
    elif 'cos' in formula:
        description += "It is a cosine wave, also showing periodic oscillations."
    elif 'exp' in formula:
        description += "It is an exponential function, showing rapid growth or decay."
    elif '^2' in formula or '**2' in formula:
        description += "It is a quadratic function, showing a parabolic shape."
    else:
        description += "It has a unique shape based on the provided function."
    return description

# Function to generate multiple questions and answers based on the graph description
def generate_qa(description):
    prompts = [
        f"The following description is given for a graph: {description} What is an important feature of this graph, and explain it in detail?",
        f"The following description is given for a graph: {description} How would you describe the behavior of the graph?",
        f"The following description is given for a graph: {description} What are the critical points of this graph?",
        f"The following description is given for a graph: {description} How does this graph change as x increases?"
    ]
    
    questions_answers = []
    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions_answers.append((prompt, answer))
    
    return questions_answers

# Load pre-trained GPT model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Get user input for the formula
user_formula = input("Enter the formula for the graph (use 'np' for numpy functions, e.g., 'np.sin(x)'): ")

# Define the range for x
x_range = (0, 2 * np.pi)

# Generate the graph
graph_buf = generate_graph(user_formula, x_range)
description = describe_graph(user_formula)

# Generate questions and answers based on the graph description
questions_answers = generate_qa(description)
for i, (question, answer) in enumerate(questions_answers):
    print(f"Question {i+1}: {question}")
    print(f"Answer {i+1}: {answer}")
    print()

# Optionally display the graph
if graph_buf:
    from PIL import Image
    image = Image.open(graph_buf)
    image.show()
