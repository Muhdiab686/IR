from transformers import pipeline

class TextGenerator:
    def __init__(self, model_name: str = "t5-small"):
        print(f"Loading text generation model: {model_name}...")
        self.generator = pipeline("text2text-generation", model=model_name)
        print("✅ Generation model loaded.")

    def answer_from_context(self, context: str, query: str, max_length: int = 150) -> str:
        prompt = f"""
        Answer the following question based only on the context provided.
        
        Context: {context[:4000]} # نقص السياق لتجنب النصوص الطويلة جدًا
        
        Question: {query}
        
        Answer:
        """
        
        generated = self.generator(prompt, max_length=max_length, num_beams=4, early_stopping=True)
        return generated[0]['generated_text']
    


