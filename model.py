from transformers import GPT2Tokenizer, GPT2ForSequenceClassification


class Matcher:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.print_model_info()

    def print_model_info(self):
        print(f"trainable params: {self.model.num_parameters()}", flush=True)

class GPTMatcher(Matcher):
    def __init__(self, base_model: str = 'gpt2'):
        self.model = GPT2ForSequenceClassification.from_pretrained(base_model)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        super().__init__(self.model, self.tokenizer)

def load_model(base_model):
    if 'gpt' in base_model:
        model = GPTMatcher(base_model)
    else:
        raise ValueError('Model not found.')
    return model.model, model.tokenizer
