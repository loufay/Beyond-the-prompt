import torch
from PIL import Image
import os
import tempfile
import base64
import io
from typing import List, Optional
from MedImageInsight.UniCLModel import build_unicl_model
from MedImageInsight.Utils.Arguments import load_opt_from_config_files
from MedImageInsight.ImageDataLoader import build_transforms
from MedImageInsight.LangEncoder import build_tokenizer
from peft import LoraConfig, get_peft_model


class MedImageInsight:
    """Wrapper class for medical image classification model."""

    def __init__(
        self,
        model_dir: str,
        vision_model_name: str,
        language_model_name: str,
    ) -> None:
        """Initialize the medical image classifier.

        Args:
            model_dir: Directory containing model files and config
            vision_model_name: Name of the vision model
            language_model_name: Name of the language model
        """
        self.model_dir = model_dir
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.model = None
        self.device = None
        self.tokenize = None
        self.preprocess = None
        self.opt = None
        
    def load_model(self, use_lora: bool = False, lora_config: Optional[dict] = None) -> None:
        """Load the model and optionally add LoRA adapters."""
        try:
            # Load configuration
            config_path = os.path.join(self.model_dir, 'config.yaml')
            self.opt = load_opt_from_config_files([config_path])

            # Add `use_return_dict` to ensure compatibility
            self.opt['use_return_dict'] = True

            # Set paths
            self.opt['LANG_ENCODER']['PRETRAINED_TOKENIZER'] = os.path.join(
                self.model_dir,
                'language_model',
                'clip_tokenizer_4.16.2'
            )
            self.opt['UNICL_MODEL']['PRETRAINED'] = os.path.join(
                self.model_dir,
                'vision_model',
                self.vision_model_name
            )

            # Initialize components
            self.preprocess = build_transforms(self.opt, False)
            self.model = build_unicl_model(self.opt)

            # Add LoRA adapters if specified
            if use_lora and lora_config:
                from peft import LoraConfig, get_peft_model
                lora_configuration = LoraConfig(**lora_config)
                self.model = get_peft_model(self.model, lora_configuration)
                self.model.config = self.opt

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Load tokenizer
            self.tokenize = build_tokenizer(self.opt['LANG_ENCODER'])
            self.max_length = self.opt['LANG_ENCODER']['CONTEXT_LENGTH']

            print(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            print("Failed to load the model:")
            raise e


    @staticmethod
    def decode_base64_image(base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image and ensure RGB format."""
        try:
            # Remove header if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]

            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert grayscale (L) or grayscale with alpha (LA) to RGB
            if image.mode in ('L', 'LA'):
                image = image.convert('RGB')

            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    def predict(self, images: List[str], labels: List[str], multilabel: bool = False) -> List[dict]:
        """Perform zero-shot classification on the input images."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not labels:
            raise ValueError("No labels provided")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Process images
            image_list = []
            for img_base64 in images:
                try:
                    img = self.decode_base64_image(img_base64)
                    image_list.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to process image: {str(e)}")

            # Run inference
            probs = self.run_inference_batch(image_list, labels, multilabel)
            probs_np = probs.cpu().numpy()
            results = []
            for prob_row in probs_np:
                # Create label-prob pairs and sort by probability
                label_probs = [(label, float(prob)) for label, prob in zip(labels, prob_row)]
                label_probs.sort(key=lambda x: x[1], reverse=True)

                # Create ordered dictionary from sorted pairs
                results.append({
                    label: prob
                    for label, prob in label_probs
                })

            return results

    def run_inference_batch(
            self,
            images: List[Image.Image],
            texts: List[str],
            multilabel: bool = False
    ) -> torch.Tensor:
        """Perform inference on batch of input images."""
        # Prepare inputs
        images = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        # Process text
        text_tokens = self.tokenize(
            texts,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        # Move text tensors to the correct device
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(image=images, text=text_tokens)
            logits_per_image = outputs[0] @ outputs[1].t() * outputs[2]

            if multilabel:
                # Use sigmoid for independent probabilities per label
                probs = torch.sigmoid(logits_per_image)
            else:
                # Use softmax for single-label classification
                probs = logits_per_image.softmax(dim=1)

        return probs

    def fine_tune(self, train_loader, val_loader, epochs=10, lr=5e-4):
        """Fine-tune the model using LoRA adapters."""
        # Use self.opt as the config reference
        config = self.opt

        # Enable training for LoRA parameters only
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for images, labels in train_loader:
                image_list = [self.decode_base64_image(img_base64) for img_base64 in images]
                image_tensors = torch.stack([self.preprocess(img) for img in image_list]).to(self.device)

                # Forward pass
                outputs = self.model(image=image_tensors)["logits"]

                # Compute loss
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                total_train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loop
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    image_list = [self.decode_base64_image(img_base64) for img_base64 in images]
                    image_tensors = torch.stack([self.preprocess(img) for img in image_list]).to(self.device)
                    outputs = self.model(image=image_tensors)["logits"]
                    labels = labels.to(self.device)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {total_train_loss/len(train_loader):.4f}, "
                f"Validation Loss = {total_val_loss/len(val_loader):.4f}")

    def save_lora(self, save_path):
        """Save LoRA parameters."""
        self.model.save_pretrained(save_path)

    def load_lora(self, lora_path):
        """Load LoRA parameters."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.to(self.device)
