"""
pixxEngine Encryption Module
=============================
Fernet (AES-128-CBC) encryption for sensitive biometric data.

CRITICAL: This module provides application-level encryption for:
- Facial embeddings (InsightFace 512-dim vectors)
- Body/ReID vectors (TransReID feature vectors)

Data is encrypted BEFORE entering the message queue and decrypted
AFTER being read by the database scribe.
"""

import base64
import logging
from typing import Union, List, Optional
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from src.detection_config import settings

# Configure logging
logger = logging.getLogger(__name__)


class VectorEncryption:
    """
    Handles encryption/decryption of biometric vectors using Fernet.
    
    Fernet guarantees that data encrypted using it cannot be manipulated
    or read without the key. It uses AES-128-CBC with HMAC-SHA256 for
    authentication.
    """
    
    def __init__(self, key: str = None):
        """
        Initialize encryption with Fernet key.
        
        Args:
            key: Base64-encoded Fernet key (32 bytes when decoded)
                 If not provided, uses FERNET_KEY from settings
        """
        self._key = key or settings.FERNET_KEY
        
        if not self._key:
            raise ValueError(
                "FERNET_KEY not configured. Generate one with:\n"
                "python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        
        try:
            # Ensure key is bytes
            if isinstance(self._key, str):
                self._key = self._key.encode()
            
            self._fernet = Fernet(self._key)
            logger.info("VectorEncryption initialized successfully")
            
        except Exception as e:
            raise ValueError(f"Invalid Fernet key: {e}")
    
    def encrypt_vector(self, vector: Union[np.ndarray, List[float]]) -> bytes:
        """
        Encrypt a feature vector.
        
        Args:
            vector: NumPy array or list of floats representing the vector
        
        Returns:
            Encrypted bytes ready for storage/transmission
        """
        try:
            # Convert to numpy array if needed
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            elif isinstance(vector, np.ndarray):
                vector = vector.astype(np.float32)
            
            # Serialize to bytes
            vector_bytes = vector.tobytes()
            
            # Encrypt
            encrypted = self._fernet.encrypt(vector_bytes)
            
            logger.debug(f"Encrypted vector of shape {vector.shape}, {len(encrypted)} bytes")
            return encrypted
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_vector(self, encrypted_data: bytes, shape: tuple = None) -> np.ndarray:
        """
        Decrypt an encrypted feature vector.
        
        Args:
            encrypted_data: Encrypted bytes
            shape: Optional tuple to reshape the vector (e.g., (512,) for face embeddings)
        
        Returns:
            Decrypted numpy array
        """
        try:
            # Decrypt
            decrypted_bytes = self._fernet.decrypt(encrypted_data)
            
            # Reconstruct numpy array
            vector = np.frombuffer(decrypted_bytes, dtype=np.float32)
            
            if shape:
                vector = vector.reshape(shape)
            
            logger.debug(f"Decrypted vector of shape {vector.shape}")
            return vector
            
        except InvalidToken:
            logger.error("Decryption failed: Invalid token (wrong key or corrupted data)")
            raise
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def encrypt_vector_b64(self, vector: Union[np.ndarray, List[float]]) -> str:
        """
        Encrypt a vector and return as base64 string (for JSON serialization).
        
        Args:
            vector: NumPy array or list of floats
        
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.encrypt_vector(vector)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_vector_b64(self, b64_data: str, shape: tuple = None) -> np.ndarray:
        """
        Decrypt a base64-encoded encrypted vector.
        
        Args:
            b64_data: Base64-encoded encrypted string
            shape: Optional shape to reshape vector
        
        Returns:
            Decrypted numpy array
        """
        encrypted = base64.b64decode(b64_data.encode('utf-8'))
        return self.decrypt_vector(encrypted, shape)


# Global encryption instance (lazy initialization)
_encryptor: Optional[VectorEncryption] = None


def get_encryptor() -> VectorEncryption:
    """Get or create the global encryptor instance."""
    global _encryptor
    if _encryptor is None:
        _encryptor = VectorEncryption()
    return _encryptor


# ============================================
# Key Generation Utility
# ============================================

def generate_fernet_key() -> str:
    """
    Generate a new Fernet encryption key.
    
    Returns:
        Base64-encoded Fernet key string
    """
    return Fernet.generate_key().decode('utf-8')


if __name__ == "__main__":
    # Utility: Generate a new key when run directly
    print("Generated Fernet Key:")
    print(generate_fernet_key())
    print("\nAdd this to your .env file as FERNET_KEY=<key>")
