# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from outlines.models import Transformers
# from outlines import generate
#
#
# # 1. Define your Pydantic model
# class Pet(BaseModel):
#     animal_type: Literal['cat', 'dog', 'bird']
#     name: str
#     age: int
#     favorite_food: str
#
# class NameYear(BaseModel):
#     name: str
#     year: int
#
# # 2. Initialize Outlines model
# model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
# model = Transformers(
#     AutoModelForCausalLM.from_pretrained(model_name),
#     AutoTokenizer.from_pretrained(model_name)
# )
#
# # 3. Create JSON generator
# pet_generator = generate.json(model, Pet)
#
#
# # 4. Define your test prompt
# prompt = "Generate a JSON object for a 2 year old cat named Whiskers who loves tuna"
#
# # 5. Generate and output
# try:
#     # Outlines already returns a validated Pet object
#     pet = pet_generator(prompt)
#
#     print("Generated successfully!")
#     print(f"Type: {type(pet)}")  # Should show <class '__main__.Pet'>
#     print(pet)
#     response = pet.model_dump_json(indent=2)
#     print(type(response))
#     print(response)
#
# except Exception as e:
#     print(f"Error: {str(e)}")