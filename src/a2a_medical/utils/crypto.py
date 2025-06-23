"""
Cryptography utilities for medical A2A security.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    FERNET = "fernet"


class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA_256 = "sha-256"
    SHA_512 = "sha-512"
    BLAKE2B = "blake2b"


@dataclass
class EncryptionResult:
    """Result of an encryption operation."""
    
    encrypted_data: bytes
    algorithm: EncryptionAlgorithm
    key_id: Optional[str] = None
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecryptionResult:
    """Result of a decryption operation."""
    
    decrypted_data: bytes
    algorithm: EncryptionAlgorithm
    key_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CryptoManager(ABC):
    """Base class for cryptographic operations in medical A2A systems."""
    
    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.keys: Dict[str, Any] = {}
        self.key_rotation_policy: Dict[str, Any] = {}
        self.encryption_stats: Dict[str, int] = {
            "encryptions": 0,
            "decryptions": 0,
            "key_generations": 0,
            "errors": 0
        }
    
    @abstractmethod
    async def encrypt(self, data: Union[str, bytes], key_id: Optional[str] = None) -> EncryptionResult:
        """Encrypt data using the specified or default key."""
        pass
    
    @abstractmethod
    async def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None) -> DecryptionResult:
        """Decrypt data using the specified or default key."""
        pass
    
    @abstractmethod
    async def generate_key(self, algorithm: EncryptionAlgorithm, key_id: Optional[str] = None) -> str:
        """Generate a new encryption key."""
        pass
    
    @abstractmethod
    async def hash_data(self, data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA_256) -> str:
        """Hash data using the specified algorithm."""
        pass
    
    def add_key(self, key_id: str, key: Any) -> None:
        """Add a key to the key store."""
        self.keys[key_id] = key
    
    def remove_key(self, key_id: str) -> None:
        """Remove a key from the key store."""
        if key_id in self.keys:
            del self.keys[key_id]
    
    def get_key(self, key_id: str) -> Optional[Any]:
        """Get a key by ID."""
        return self.keys.get(key_id)
    
    def update_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update encryption statistics."""
        if stat_name in self.encryption_stats:
            self.encryption_stats[stat_name] += increment
    
    def get_encryption_stats(self) -> Dict[str, int]:
        """Get encryption statistics."""
        return self.encryption_stats.copy()


class SimpleCryptoManager(CryptoManager):
    """Simple cryptography manager implementation."""
    
    def __init__(self, manager_id: str):
        super().__init__(manager_id)
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM
        self.default_key_id = "default"
        
        # Generate default key if cryptography is available
        if CRYPTOGRAPHY_AVAILABLE:
            self._generate_default_key()
    
    def _generate_default_key(self) -> None:
        """Generate a default encryption key."""
        if CRYPTOGRAPHY_AVAILABLE:
            key = Fernet.generate_key()
            self.keys[self.default_key_id] = key
    
    async def encrypt(self, data: Union[str, bytes], key_id: Optional[str] = None) -> EncryptionResult:
        """Encrypt data using Fernet encryption."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        key_id = key_id or self.default_key_id
        key = self.get_key(key_id)
        
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        try:
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt using Fernet
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data)
            
            self.update_stats("encryptions")
            
            return EncryptionResult(
                encrypted_data=encrypted_data,
                algorithm=EncryptionAlgorithm.FERNET,
                key_id=key_id
            )
        
        except Exception as e:
            self.update_stats("errors")
            raise RuntimeError(f"Encryption failed: {str(e)}")
    
    async def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None) -> DecryptionResult:
        """Decrypt data using Fernet decryption."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        key_id = key_id or self.default_key_id
        key = self.get_key(key_id)
        
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        try:
            # Decrypt using Fernet
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            self.update_stats("decryptions")
            
            return DecryptionResult(
                decrypted_data=decrypted_data,
                algorithm=EncryptionAlgorithm.FERNET,
                key_id=key_id
            )
        
        except Exception as e:
            self.update_stats("errors")
            raise RuntimeError(f"Decryption failed: {str(e)}")
    
    async def generate_key(self, algorithm: EncryptionAlgorithm, key_id: Optional[str] = None) -> str:
        """Generate a new encryption key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        if algorithm == EncryptionAlgorithm.FERNET:
            key = Fernet.generate_key()
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        key_id = key_id or f"key_{secrets.token_hex(8)}"
        self.keys[key_id] = key
        self.update_stats("key_generations")
        
        return key_id
    
    async def hash_data(self, data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA_256) -> str:
        """Hash data using the specified algorithm."""
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == HashAlgorithm.SHA_256:
            hash_obj = hashlib.sha256()
        elif algorithm == HashAlgorithm.SHA_512:
            hash_obj = hashlib.sha512()
        elif algorithm == HashAlgorithm.BLAKE2B:
            hash_obj = hashlib.blake2b()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj.update(data)
        return hash_obj.hexdigest()


class AdvancedCryptoManager(CryptoManager):
    """Advanced cryptography manager with multiple algorithms."""
    
    def __init__(self, manager_id: str):
        super().__init__(manager_id)
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM
        
        # Generate default keys
        if CRYPTOGRAPHY_AVAILABLE:
            self._generate_default_keys()
    
    def _generate_default_keys(self) -> None:
        """Generate default encryption keys."""
        # Generate AES key
        aes_key = os.urandom(32)  # 256-bit key
        self.keys["aes_default"] = aes_key
        
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        self.keys["rsa_private"] = private_key
        self.keys["rsa_public"] = public_key
    
    async def encrypt(self, data: Union[str, bytes], key_id: Optional[str] = None) -> EncryptionResult:
        """Encrypt data using AES-256-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate random IV
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        # Get encryption key
        key = self.get_key(key_id or "aes_default")
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        try:
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
            encryptor = cipher.encryptor()
            
            # Encrypt data
            encrypted_data = encryptor.update(data) + encryptor.finalize()
            tag = encryptor.tag
            
            self.update_stats("encryptions")
            
            return EncryptionResult(
                encrypted_data=encrypted_data,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_id=key_id or "aes_default",
                iv=iv,
                tag=tag
            )
        
        except Exception as e:
            self.update_stats("errors")
            raise RuntimeError(f"Encryption failed: {str(e)}")
    
    async def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None, 
                     iv: Optional[bytes] = None, tag: Optional[bytes] = None) -> DecryptionResult:
        """Decrypt data using AES-256-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        if not iv or not tag:
            raise ValueError("IV and tag are required for AES-GCM decryption")
        
        # Get decryption key
        key = self.get_key(key_id or "aes_default")
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        try:
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            
            # Decrypt data
            decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            self.update_stats("decryptions")
            
            return DecryptionResult(
                decrypted_data=decrypted_data,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_id=key_id or "aes_default"
            )
        
        except Exception as e:
            self.update_stats("errors")
            raise RuntimeError(f"Decryption failed: {str(e)}")
    
    async def generate_key(self, algorithm: EncryptionAlgorithm, key_id: Optional[str] = None) -> str:
        """Generate a new encryption key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            key = os.urandom(32)  # 256-bit key
        elif algorithm == EncryptionAlgorithm.RSA_2048:
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
        elif algorithm == EncryptionAlgorithm.RSA_4096:
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        key_id = key_id or f"key_{secrets.token_hex(8)}"
        self.keys[key_id] = key
        self.update_stats("key_generations")
        
        return key_id
    
    async def hash_data(self, data: Union[str, bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA_256) -> str:
        """Hash data using the specified algorithm."""
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == HashAlgorithm.SHA_256:
            hash_obj = hashlib.sha256()
        elif algorithm == HashAlgorithm.SHA_512:
            hash_obj = hashlib.sha512()
        elif algorithm == HashAlgorithm.BLAKE2B:
            hash_obj = hashlib.blake2b()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj.update(data)
        return hash_obj.hexdigest()
    
    async def sign_data(self, data: Union[str, bytes], key_id: str = "rsa_private") -> bytes:
        """Sign data using RSA private key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        private_key = self.get_key(key_id)
        if not private_key:
            raise ValueError(f"Private key {key_id} not found")
        
        # Sign the data
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    async def verify_signature(self, data: Union[str, bytes], signature: bytes, 
                             key_id: str = "rsa_public") -> bool:
        """Verify data signature using RSA public key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        public_key = self.get_key(key_id)
        if not public_key:
            raise ValueError(f"Public key {key_id} not found")
        
        try:
            # Verify the signature
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
