import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
from eagle.model.choices import linear_tree_len_6

def generate():
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # device_map="auto",
    )
    model.eval()

    inputs = model.tokenizer([prompt], return_tensors="pt")

    torch.save(inputs.input_ids, "prompt_input_ids.pt")

    # input_idsf
    # tensor([[128000,     40,   4510,    279,   7438,    315,   2324,    374]])

    output_ids = model.eagenerate(
        inputs.input_ids,
        temperature=0.0,
        top_k=1,
        max_new_tokens=18,
        tree_choices=linear_tree_len_6)
    print("Done!")


if __name__ == "__main__":
    base_model_path = "/home/ubuntu/xiufen/models/llama-3.1-405b-4l"
    EAGLE_model_path = "/home/ubuntu/xiufen/models/eagle"
    prompt = "I believe the meaning of life is"
    # neuron_spec_len = 6
    generate()