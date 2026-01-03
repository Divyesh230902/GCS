"""
Model Configuration for GCS
Allows easy switching between different models and handles authentication
"""

import os
from typing import Dict, Any

# Model configurations
MODEL_CONFIGS = {
    "mamba-1.4b": {
        "model_name": "state-spaces/mamba-1.4b-hf",
        "model_type": "causal_lm",
        "requires_auth": True,
        "description": "Mamba 1.4B state-space model"
    },
    "mamba-130m": {
        "model_name": "state-spaces/mamba-130m-hf", 
        "model_type": "causal_lm",
        "requires_auth": True,
        "description": "Mamba 130M state-space model"
    },
    "dialo-gpt-small": {
        "model_name": "microsoft/DialoGPT-small",
        "model_type": "causal_lm", 
        "requires_auth": False,
        "description": "DialoGPT Small (fallback)"
    },
    "gpt2": {
        "model_name": "gpt2",
        "model_type": "causal_lm",
        "requires_auth": False,
        "description": "GPT-2 (public model)"
    },
    "rule-based": {
        "model_name": None,
        "model_type": "rule_based",
        "requires_auth": False,
        "description": "Rule-based fallback (no neural model)"
    }
}

def get_model_config(model_key: str = "rule-based") -> Dict[str, Any]:
    """
    Get model configuration
    
    Args:
        model_key: Key for the model configuration
        
    Returns:
        Model configuration dictionary
    """
    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model key: {model_key}")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return MODEL_CONFIGS["rule-based"]
    
    return MODEL_CONFIGS[model_key]

def check_huggingface_auth() -> bool:
    """
    Check if Hugging Face authentication is available
    
    Returns:
        True if authenticated, False otherwise
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try to access a gated model
        api.model_info("state-spaces/mamba-1.4b-hf")
        return True
    except Exception:
        return False

def get_available_models() -> Dict[str, str]:
    """
    Get list of available models based on authentication status
    
    Returns:
        Dictionary of available models and their descriptions
    """
    available = {}
    has_auth = check_huggingface_auth()
    
    for key, config in MODEL_CONFIGS.items():
        if not config["requires_auth"] or has_auth:
            available[key] = config["description"]
    
    return available

def setup_huggingface_auth():
    """
    Instructions for setting up Hugging Face authentication
    """
    print("=" * 60)
    print("HUGGING FACE AUTHENTICATION SETUP")
    print("=" * 60)
    print("\nTo use Mamba state-space models, you need to:")
    print("1. Create a Hugging Face account at https://huggingface.co")
    print("2. Get an access token from https://huggingface.co/settings/tokens")
    print("3. Run one of these commands:")
    print("   - `huggingface-cli login`")
    print("   - `export HUGGINGFACE_HUB_TOKEN=your_token_here`")
    print("   - Or set the token in your environment")
    print("\nAlternatively, the system will use rule-based fallback models.")
    print("=" * 60)

if __name__ == "__main__":
    print("Available models:")
    available = get_available_models()
    for key, desc in available.items():
        print(f"  {key}: {desc}")
    
    if not check_huggingface_auth():
        print("\n⚠️  No Hugging Face authentication detected")
        setup_huggingface_auth()
    else:
        print("\n✅ Hugging Face authentication available")
