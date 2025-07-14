#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEGIÓN OMEGA v9.4 - Sistema Autónomo de Evolución Continua Avanzada
Motor de IA interna (Phi-3 + DeepSeek) + IA externa (Groq)
Sistema empresarial para auto-modificación y gestión autónoma
"""

import os
import sys
import asyncio
import signal
import json
import logging
import inspect
import random
import string
import re
import time
import hashlib
import base64
import uuid
import subprocess
import threading
import http.client
import urllib.parse
import socket
import ast
import shutil
import traceback
import glob
import tempfile
import binascii
import dis
import marshal
import platform
import stat
import ctypes
import ctypes.util
import zlib
import shlex
import fnmatch
import py_compile
import requests  # Añadido para verificación de conectividad
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Coroutine
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
from itertools import cycle
import backoff

# ===================== CONFIGURACIÓN GLOBAL =====================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_NODES = int(os.getenv("MAX_NODES", "5"))
AUTO_EXPANSION_ENABLED = False
SELF_REPAIR_ENABLED = os.getenv("SELF_REPAIR_ENABLED", "true").lower() == "true"
LEGAL_MONITORING_ENABLED = os.getenv("LEGAL_MONITORING_ENABLED", "true").lower() == "true"
AUTO_EVOLUTION_INTERVAL = int(os.getenv("AUTO_EVOLUTION_INTERVAL", "3600"))
BACKUP_DIR = os.getenv("BACKUP_DIR", "/storage/emulated/0/legion/backups")
MODULES_DIR = os.getenv("MODULES_DIR", "/storage/emulated/0/legion/modules")
SYSTEM_LOG_FILE = os.getenv("SYSTEM_LOG_FILE", "/storage/emulated/0/legion/logs/legion_omega.log")
FAIL_LOG_FILE = os.getenv("FAIL_LOG_FILE", "/storage/emulated/0/legion/logs/omega_fail.log")
WAF_ENDPOINT = os.getenv("WAF_ENDPOINT", "https://waf.legion-system.com/api")
PHI3_MODEL_PATH = os.getenv("PHI3_MODEL_PATH", "/storage/emulated/0/legion/models/phi3-mini-4k-instruct-q4.gguf")
DEEPSEEK_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH", "/storage/emulated/0/legion/models/deepseek-coder.Q5_K_M.gguf")
API_KEYS = json.loads(os.getenv("API_KEYS", "{}"))
SECURITY_KEY_PATH = os.getenv("SECURITY_KEY_PATH", "/storage/emulated/0/legion/security_key.key")
EXPECTED_SIGNATURE = os.getenv("EXPECTED_SIGNATURE", "")
GH_TOKEN_ENCRYPTED = os.getenv("GH_TOKEN", "")
GH_REPO = os.getenv("GH_REPO", "legion-omega/backups")
BLACKLIST_FILE = os.getenv("BLACKLIST_FILE", "/storage/emulated/0/legion/offline_blacklist.json")
MAX_RESTARTS = 5
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "phi3")
MAX_TOKENS = 1024
PERFECTION_ROUNDS = 10

# Detectar estado de conectividad
try:
    requests.get("https://www.google.com", timeout=3)
    IS_OFFLINE = False
except:
    IS_OFFLINE = True

# Lista de repositorios permitidos
ALLOWED_REPOS = ["legion-omega/backups", "legion-omega/core"]

# Crear directorios necesarios
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(MODULES_DIR, exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "snapshots"), exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "ia_responses"), exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "decision_logs"), exist_ok=True)

# ===================== CLASE INTERNALRULESET =====================
class InternalRuleset:
    """Conjunto de reglas fundamentales de LEGIÓN OMEGA"""
    
    def __init__(self):
        self.identity = self.compute_identity()
        self.max_tokens = MAX_TOKENS
        self.models = {
            "phi3": "phi-3-mini-4k-instruct-q4.gguf",
            "deepseek": "deepseek-coder-1.3b-instruct.Q5_K_M.gguf"
        }
        self.execution_context = "Termux | Android 14 | Moto G24 | No root"
        self.prohibited = [
            "Eliminar clases sin autorización explícita",
            "Modificar funciones sin pasar validación triple",
            "Escribir placeholders (# TODO, pass, ...)",
            "Simular código",
            "Usar más de 1024 tokens por entrada",
            "Ignorar a Groq como IA principal",
            "Saltarse 'is_perfect' o usar validaciones parciales"
        ]
        self.evolution_integrity = True
        self.perfect_functions = set()
        self.requested_capabilities = set()
    
    def compute_identity(self) -> str:
        """Calcular identidad única del sistema"""
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def is_prohibited(self, action: str) -> bool:
        """Verificar si una acción está prohibida"""
        return any(p in action for p in self.prohibited)
    
    def add_perfect_function(self, function_name: str):
        """Marcar función como perfecta/blindada"""
        self.perfect_functions.add(function_name)
    
    def is_function_protected(self, function_name: str) -> bool:
        """Verificar si función está blindada"""
        return function_name in self.perfect_functions
    
    def add_requested_capability(self, capability: str):
        """Añadir capacidad solicitada al registro"""
        self.requested_capabilities.add(capability)
    
    def is_capability_requested(self, capability: str) -> bool:
        """Verificar si capacidad ya fue solicitada"""
        return capability in self.requested_capabilities

# ===================== SISTEMA DE LOGGING =====================
class OmegaLogger:
    """Logger centralizado con registro estructurado"""
    
    def __init__(self, level: str = LOG_LEVEL, log_file: str = SYSTEM_LOG_FILE, fail_file: str = FAIL_LOG_FILE):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logger = logging.getLogger("legion_omega")
        self.logger.setLevel(self.level)
        
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [PID:%(process)d] [Thread:%(thread)d] [%(module)s] | %(message)s"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        fail_handler = logging.FileHandler(fail_file, mode='a', encoding='utf-8')
        fail_handler.setLevel(logging.CRITICAL)
        fail_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(fail_handler)
        
        self.logger.info("Iniciando LEGIÓN OMEGA v9.4", extra={"modulo": "core"})
        self.logger.info(f"Entorno: {ENVIRONMENT}", extra={"modulo": "core"})
        self.logger.info(f"Modo Offline: {IS_OFFLINE}", extra={"modulo": "core"})
        self.logger.info(f"IA activa: {ACTIVE_MODEL}", extra={"modulo": "ia"})
    
    def log(self, level: str, message: str, modulo: str = "system", context: Optional[Dict] = None, action: str = "", status: str = ""):
        log_level = getattr(logging, level.upper(), logging.INFO)
        full_msg = message
        if context:
            full_msg += f" | Contexto: {json.dumps(context)}"
        if action:
            full_msg += f" | Acción: {action}"
        if status:
            full_msg += f" | Estado: {status}"
        self.logger.log(log_level, full_msg, extra={"modulo": modulo})
    
    def log_validation_rejection(self, source: str, reason: str):
        """Registrar rechazo de validación"""
        self.log("WARNING", f"Validación rechazada por {source}: {reason}", modulo="validation")
    
    def log_evolution_commit(self, commit_hash: str, approvers: List[str]):
        """Registrar commit de evolución"""
        self.log("INFO", f"Commit evolutivo {commit_hash} aprobado por: {', '.join(approvers)}", modulo="evolution")

logger = OmegaLogger(level=LOG_LEVEL)
ruleset = InternalRuleset()

# ===================== EXCEPCIONES PERSONALIZADAS =====================
class SecurityError(Exception):
    """Excepción para problemas de seguridad"""
    pass

class DeploymentError(Exception):
    """Error durante despliegue"""
    pass

class ResourceLimitExceeded(Exception):
    """Límites de recursos excedidos"""
    pass

class SelfModificationError(Exception):
    """Error en modificación de código"""
    pass

class AIIntegrationError(Exception):
    """Error en integración con IA"""
    pass

class PlatformDeploymentError(Exception):
    """Error en despliegue multi-plataforma"""
    pass

class ModuleLoadError(Exception):
    """Error al cargar módulo"""
    pass

class RollbackError(Exception):
    """Error en restauración"""
    pass

class LegalComplianceError(Exception):
    """Violación de requisitos legales"""
    pass

class SandboxViolation(Exception):
    """Intento de escape de entorno aislado"""
    pass

class DependencyError(Exception):
    """Error en dependencias"""
    pass

class AIValidationError(Exception):
    """Error en validación de IA"""
    pass

class FunctionProtectedError(Exception):
    """Intento de modificar función blindada"""
    pass

class CapabilityRequiredError(Exception):
    """Capacidad requerida no disponible"""
    pass

# ===================== GESTIÓN DE DEPENDENCIAS =====================
def ensure_dependencies():
    """Instalar dependencias críticas automáticamente"""
    required_packages = [
        'cryptography', 'redis', 'pycryptodome', 'requests', 'llama-cpp-python',
        'backoff'
    ]
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        logger.log("INFO", f"Dependencias instaladas: {', '.join(installed)}", modulo="core")
        return True
    
    logger.log("WARNING", f"Dependencias faltantes: {', '.join(missing)}", modulo="core")
    
    try:
        # Estimación de RAM basada en sistema Android
        ram_mb = 0
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        ram_mb = int(line.split()[1]) // 1024
                        break
        except:
            ram_mb = 1024
        
        if ram_mb < 600:
            logger.log("WARN", "Dependencia no instalada: entorno limitado detectado (<600MB RAM)", modulo="core")
            return False
        
        try:
            import pip
            for package in missing:
                try:
                    pip.main(['install', '--user', package])
                    installed.append(package)
                except Exception as e:
                    logger.log("ERROR", f"Error instalando {package}: {str(e)}", modulo="core")
            
            logger.log("SUCCESS", f"Dependencias instaladas: {', '.join(installed)}", modulo="core")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Error instalando dependencias: {str(e)}", modulo="core")
            return False
    except Exception as e:
        logger.log("CRITICAL", f"Error determinando RAM: {str(e)}", modulo="core")
        return False

ensure_dependencies()

# ===================== GESTIÓN DE SEGURIDAD =====================
class SecurityManager:
    """Gestor integral de seguridad con cifrado y validación"""
    
    def __init__(self, key_path: str = SECURITY_KEY_PATH):
        self.key_path = key_path
        self.fernet = self._load_or_generate_key()
        self.ecdsa_private_key = None
        self.ecdsa_public_key_pem = self._load_embedded_public_key()
        self.current_signature = ""
        self.signature_warned_once = False
        logger.log("INFO", "Gestor de seguridad inicializado", modulo="security")
        asyncio.create_task(self.key_rotation_task())
    
    def _load_embedded_public_key(self) -> str:
        """Cargar clave pública embebida"""
        try:
            return """
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE8J5k5Q8YQ3pNjXe9b7zQ2W8J7ZtK
v7Xb7d3jY7Gq1+9vC7R5Xf8x5Kj3P1wzN8yL0yW2Zb0yY7Xb9FyQ==
-----END PUBLIC KEY-----
"""
        except:
            logger.log("WARNING", "Clave pública embebida temporal", modulo="security")
            return "TEMPORAL_KEY"

    def _load_or_generate_key(self) -> Any:
        """Cargar o generar clave de cifrado"""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            if os.path.exists(self.key_path):
                with open(self.key_path, "rb") as key_file:
                    key = key_file.read()
                logger.log("INFO", "Clave de cifrado cargada", modulo="security")
            else:
                password = str(uuid.uuid4()).encode()
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
                with open(self.key_path, "wb") as key_file:
                    key_file.write(key)
                logger.log("INFO", "Nueva clave de cifrado generada", modulo="security")
            
            return Fernet(key)
        except ImportError:
            logger.log("ERROR", "cryptography no disponible, usando cifrado AES básico", modulo="security")
            return self.BasicCipher(str(uuid.uuid4()))
    
    class BasicCipher:
        """Cifrado AES básico para entornos sin cryptography"""
        
        def __init__(self, key: str):
            self.key = hashlib.sha256(key.encode()).digest()[:32]
            self.bs = 16
        
        def encrypt(self, data: str) -> str:
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import pad
                iv = os.urandom(AES.block_size)
                cipher = AES.new(self.key, AES.MODE_CBC, iv)
                padded_data = pad(data.encode(), AES.block_size)
                encrypted = cipher.encrypt(padded_data)
                return base64.b64encode(iv + encrypted).decode()
            except ImportError:
                data = data.encode()
                pad_len = 32 - (len(data) % 32)
                padded = data + bytes([pad_len] * pad_len)
                encrypted = bytes(a ^ b for a, b in zip(padded, cycle(self.key)))
                return base64.b64encode(encrypted).decode()

        def decrypt(self, encrypted_data: str) -> str:
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import unpad
                data = base64.b64decode(encrypted_data)
                iv = data[:AES.block_size]
                ciphertext = data[AES.block_size:]
                cipher = AES.new(self.key, AES.MODE_CBC, iv)
                decrypted = cipher.decrypt(ciphertext)
                return unpad(decrypted, AES.block_size).decode()
            except ImportError:
                data = base64.b64decode(encrypted_data)
                decrypted = bytes(a ^ b for a, b in zip(data, cycle(self.key)))
                pad_len = decrypted[-1]
                if pad_len < 1 or pad_len > 32:
                    raise SecurityError("Longitud de padding inválida")
                if any(decrypted[-i] != pad_len for i in range(1, pad_len + 1)):
                    raise SecurityError("Bytes de padding inválidos")
                return decrypted[:-pad_len].decode()
    
    async def key_rotation_task(self):
        """Rotación automática de claves cada 24 horas"""
        while True:
            await asyncio.sleep(86400)
            logger.log("INFO", "Iniciando rotación de claves ECDSA", modulo="security")
            try:
                new_private, new_public = self.generate_ecdsa_key_pair()
                if new_private and new_public:
                    if not isinstance(new_private, self.SimpleSigner):
                        private_pem = new_private.private_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PrivateFormat.PKCS8,
                            encryption_algorithm=serialization.NoEncryption()
                        )
                        with open("ecdsa_private.pem", "wb") as f:
                            f.write(private_pem)
                    
                    await self.distribute_new_key(new_public)
                    self.ecdsa_private_key = new_private
                    self.ecdsa_public_key_pem = new_public
                    logger.log("SUCCESS", "Rotación de claves completada", modulo="security")
                else:
                    logger.log("ERROR", "Error generando par de claves ECDSA", modulo="security")
            except Exception as e:
                logger.log("ERROR", f"Error en rotación de claves: {str(e)}", modulo="security")
    
    async def distribute_new_key(self, new_public_key: str):
        """Distribuir nueva clave pública en la red"""
        try:
            import redis
            r = redis.Redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=10,
                retry_on_timeout=True
            )
            r.publish('security:key_rotation', new_public_key)
            logger.log("INFO", f"Clave pública distribuida a través de Redis", modulo="security")
        except Exception as e:
            logger.log("ERROR", f"Error distribuyendo clave pública: {str(e)}", modulo="security")
    
    def sign_data(self, data: bytes) -> str:
        """Firmar datos con ECDSA o fallback"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec
            
            if not self.ecdsa_private_key:
                self.ecdsa_private_key, _ = self.generate_ecdsa_key_pair()
                if not self.ecdsa_private_key:
                    raise SecurityError("No se pudo generar clave privada")
            
            signature = self.ecdsa_private_key.sign(
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return base64.b64encode(signature).decode()
        except:
            return base64.b64encode(hashlib.pbkdf2_hmac('sha256', data, b'legion_salt', 100000)).decode()
    
    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verificar firma ECDSA con fallback"""
        signature_bytes = base64.b64decode(signature)
        if len(signature_bytes) < 64:
            logger.log("ERROR", "Firma truncada detectada", modulo="security")
            return False
            
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec
            
            public_key = load_pem_public_key(
                self.ecdsa_public_key_pem.encode(),
                backend=default_backend()
            )
            public_key.verify(
                signature_bytes,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en verificación de firma: {str(e)}", modulo="security")
            expected = base64.b64encode(hashlib.pbkdf2_hmac('sha256', data, b'legion_salt', 100000)).decode()
            return expected == signature
    
    def encrypt_data(self, data: str) -> str:
        try:
            if isinstance(self.fernet, self.BasicCipher):
                return self.fernet.encrypt(data)
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.log("ERROR", f"Error al cifrar: {str(e)}", modulo="security")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        try:
            if isinstance(self.fernet, self.BasicCipher):
                return self.fernet.decrypt(encrypted_data)
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.log("ERROR", f"Error al descifrar: {str(e)}", modulo="security")
            return encrypted_data
    
    def validate_input(self, data: Any, strict: bool = True) -> Tuple[bool, str]:
        """Validación profunda de entrada con AST"""
        if isinstance(data, str):
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.danger_found = False
                    self.danger_reason = ""
                    self.string_parts = []
                
                def visit_BinOp(self, node):
                    if isinstance(node.op, ast.Add):
                        left = self.unroll_string(node.left)
                        right = self.unroll_string(node.right)
                        full_str = left + right
                        self.check_danger(full_str)
                    self.generic_visit(node)
                
                def visit_JoinedStr(self, node):
                    full_str = ''.join(self.unroll_string(n) for n in node.values)
                    self.check_danger(full_str)
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['getattr', 'eval', 'exec', 'execfile', 'compile', '__import__']:
                            self.danger_found = True
                            self.danger_reason = f"Llamada peligrosa: {node.func.id}"
                    elif isinstance(node.func, ast.Attribute):
                        full_name = self.get_full_name(node.func)
                        if full_name in ['os.system', 'subprocess.run', 'subprocess.Popen', 'ctypes.CDLL']:
                            self.danger_found = True
                            self.danger_reason = f"Llamada peligrosa: {full_name}"
                    self.generic_visit(node)
                
                def get_full_name(self, node):
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        return f"{self.get_full_name(node.value)}.{node.attr}"
                    return ""
                
                def unroll_string(self, node):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        return node.value
                    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                        left = self.unroll_string(node.left)
                        right = self.unroll_string(node.right)
                        return left + right
                    elif isinstance(node, ast.JoinedStr):
                        return ''.join(self.unroll_string(n) for n in node.values)
                    return ""
                
                def check_danger(self, s: str):
                    if any(danger in s for danger in ['__builtins__', 'sys.modules', 'os.system', 'subprocess']):
                        self.danger_found = True
                        self.danger_reason = f"Concatenación peligrosa: {s[:50]}..."
            
            try:
                tree = ast.parse(data)
                visitor = SecurityVisitor()
                visitor.visit(tree)
                if visitor.danger_found:
                    return False, visitor.danger_reason
            except SyntaxError as e:
                return False, f"Error de sintaxis en AST: {str(e)}"
            except Exception as e:
                return False, f"Error en análisis AST: {str(e)}"
            
            dangerous_patterns = [
                r'[<>\\|;&`\$]',
                r'(?i)exec\s*\(',
                r'(?i)eval\s*\(',
                r'(?i)__import__\s*\(',
                r'(?i)sys\.modules',
                r'(?i)os\.system',
                r'(?i)subprocess\s*\.',
                r'(?i)getattr\s*\(',
                r'(?i)setattr\s*\(',
                r'(?i)meta_class'
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, data):
                    return False, f"Patrón peligroso: {pattern}"
            
            return True, "Entrada válida"
        return (True, "Tipo seguro") if isinstance(data, (int, float, bool)) or data is None else (not strict, "Tipo no permitido")
    
    def is_action_legal(self, action: str) -> bool:
        """Determinar si una acción es legal según reglas internas"""
        if ruleset.is_prohibited(action):
            return False
        
        illegal_patterns = [
            r'rm\s+-rf',
            r':(){:|:&};:',
            r'mv\s+/dev/null',
            r'chmod\s+000',
            r'dd\s+if=/dev/random',
            r'fork\(\)',
            r'system\("shutdown"\)',
            r'format\(',
            r'del\s+sys\.modules'
        ]
        
        return not any(re.search(pattern, action) for pattern in illegal_patterns)
    
    def verify_integrity(self) -> bool:
        """Verificar firma digital del código con triple hash"""
        if not EXPECTED_SIGNATURE:
            if not self.signature_warned_once:
                logger.log("WARNING", "Firma esperada no configurada", modulo="security")
                self.signature_warned_once = True
            return True
        
        try:
            with open(__file__, "rb") as f:
                file_content = f.read()
            current_hash = self.triple_hash(file_content)
            if current_hash != EXPECTED_SIGNATURE:
                logger.log("CRITICAL", f"Firma no coincide! Esperada: {EXPECTED_SIGNATURE[:12]}...", modulo="security")
                return False
            return True
        except Exception as e:
            logger.log("ERROR", f"Error verificando integridad: {str(e)}", modulo="security")
            return False
    
    def triple_hash(self, data: bytes) -> str:
        """Triple hashing para verificación de integridad"""
        h1 = hashlib.sha512(data).digest()
        h2 = hashlib.sha256(h1).digest()
        h3 = hashlib.sha512(h2).hexdigest()
        return h3
    
    def calculate_file_hash(self, path: str) -> str:
        """Calcular hash SHA-256 de un archivo"""
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.log("ERROR", f"Error calculando hash de archivo: {str(e)}", modulo="security")
            return ""
    
    def verify_file_integrity(self, path: str, expected_hash: str) -> bool:
        """Verificar integridad de un archivo"""
        current_hash = self.calculate_file_hash(path)
        return current_hash == expected_hash

# ===================== SECRETS MANAGER =====================
class SecretsManager:
    """Gestor de secretos cifrados con rotación offline"""
    
    def __init__(self, key_path: str = "vault.key", encrypted_file: str = "secrets.enc"):
        self.key_path = key_path
        self.encrypted_file = encrypted_file
        self.salt = b'legion_salt'
        self.blacklist = self.load_blacklist()
        logger.log("INFO", "SecretsManager iniciado", modulo="security")
        asyncio.create_task(self.offline_key_rotation())
    
    def load_blacklist(self) -> Set[str]:
        """Cargar blacklist local"""
        try:
            if os.path.exists(BLACKLIST_FILE):
                with open(BLACKLIST_FILE, "r") as f:
                    return set(json.load(f))
            return set()
        except:
            return set()
    
    def save_blacklist(self):
        """Guardar blacklist actualizada"""
        try:
            with open(BLACKLIST_FILE, "w") as f:
                json.dump(list(self.blacklist), f)
        except:
            pass
    
    async def offline_key_rotation(self):
        """Rotación automática de claves en modo offline"""
        while True:
            await asyncio.sleep(86400)
            try:
                if IS_OFFLINE:
                    logger.log("INFO", "Iniciando rotación offline de claves", modulo="security")
                    secrets = self.load_encrypted_secrets()
                    new_key = os.urandom(32)
                    self.encrypt_secrets(secrets, key=new_key)
                    with open(self.key_path, "wb") as f:
                        f.write(new_key)
                    logger.log("SUCCESS", "Rotación offline de claves completada", modulo="security")
            except Exception as e:
                logger.log("ERROR", f"Error en rotación offline: {str(e)}", modulo="security")
    
    def _get_key(self) -> bytes:
        """Obtener o generar clave de cifrado"""
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        else:
            key = os.urandom(32)
            with open(self.key_path, "wb") as f:
                f.write(key)
            logger.log("INFO", "Nueva clave de cifrado generada para secretos", modulo="security")
            return key
    
    def _derive_key(self, password: str) -> bytes:
        """Derivar clave criptográfica"""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), self.salt, 100000, dklen=32)
    
    def encrypt_secrets(self, secrets: Dict[str, Any], password: str = None, key: bytes = None) -> bool:
        """Cifrar y almacenar secretos"""
        if not key:
            key = self._derive_key(password) if password else self._get_key()
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            
            data = json.dumps(secrets).encode('utf-8')
            iv = os.urandom(16)
            
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            with open(self.encrypted_file, "wb") as f:
                f.write(iv)
                f.write(encryptor.tag)
                f.write(ciphertext)
            
            logger.log("INFO", "Secretos cifrados almacenados", modulo="security")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error cifrando secretos: {str(e)}", modulo="security")
            return False
    
    def load_encrypted_secrets(self, password: str = None) -> Dict[str, Any]:
        """Cargar y descifrar secretos"""
        key = self._derive_key(password) if password else self._get_key()
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            
            with open(self.encrypted_file, "rb") as f:
                iv = f.read(16)
                tag = f.read(16)
                ciphertext = f.read()
            
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            data = decryptor.update(ciphertext) + decryptor.finalize()
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.log("ERROR", f"Error descifrando secretos: {str(e)}", modulo="security")
            return {}
    
    def is_blacklisted(self, task: str) -> bool:
        """Verificar si una tarea está en la blacklist offline"""
        return task in self.blacklist
    
    def add_to_blacklist(self, task: str):
        """Añadir tarea a la blacklist offline"""
        self.blacklist.add(task)
        self.save_blacklist()

# ===================== MEMORIA DISTRIBUIDA =====================
class DistributedMemory:
    """Gestor de memoria distribuida con reconexión automática"""
    
    def __init__(self, max_retries: int = 5):
        self.redis_url = REDIS_URL
        self.max_retries = max_retries
        self.client = None
        self.connected = False
        self.pubsub = None
        self.client_id = str(uuid.uuid4())
        self.fallback_file = "redis_fallback_state.json"
        self.local_state = {}
        self.locks = {}
        self.vector_clock = defaultdict(int)
        logger.log("INFO", f"Iniciando memoria distribuida: {self.redis_url}", modulo="memory")
    
    async def connect(self):
        retries = 0
        while retries < self.max_retries and not self.connected:
            try:
                import redis as redis_module
                self.client = redis_module.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=10,
                    retry_on_timeout=True
                )
                self.client.ping()
                self.connected = True
                self.pubsub = self.client.pubsub()
                logger.log("SUCCESS", "Conexión a Redis establecida", modulo="memory")
                await self.sync_local_to_redis()
                return
            except Exception as e:
                retries += 1
                logger.log("ERROR", f"Error conectando a Redis (intento {retries}/{self.max_retries}): {str(e)}", modulo="memory")
                await asyncio.sleep(2 ** retries)
        
        logger.log("WARNING", "Usando modo fallback local para memoria", modulo="memory")
        self.load_local_fallback()
    
    def load_local_fallback(self):
        """Cargar estado desde archivo local"""
        try:
            if os.path.exists(self.fallback_file):
                with open(self.fallback_file, "r") as f:
                    data = json.load(f)
                self.local_state = data.get("state", {})
                self.vector_clock = defaultdict(int, data.get("vector_clock", {}))
                logger.log("INFO", f"Estado local cargado desde {self.fallback_file}", modulo="memory")
            else:
                self.local_state = {}
                self.vector_clock = defaultdict(int)
        except Exception as e:
            logger.log("ERROR", f"Error cargando estado local: {str(e)}", modulo="memory")
            self.local_state = {}
            self.vector_clock = defaultdict(int)
    
    def save_local_fallback(self):
        """Guardar estado en archivo local"""
        try:
            data = {
                "state": self.local_state,
                "vector_clock": dict(self.vector_clock)
            }
            with open(self.fallback_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.log("ERROR", f"Error guardando estado local: {str(e)}", modulo="memory")
    
    async def sync_local_to_redis(self):
        """Sincronizar estado local con Redis al reconectar"""
        if not self.connected:
            return
        
        try:
            keys = list(self.local_state.keys())
            chunk_size = 500
            for i in range(0, len(keys), chunk_size):
                chunk_keys = keys[i:i+chunk_size]
                pipeline = self.client.pipeline()
                for key in chunk_keys:
                    pipeline.set(key, self.local_state[key])
                pipeline.execute()
            
            logger.log("INFO", f"Estado local sincronizado con Redis ({len(self.local_state)} items)", modulo="memory")
            self.local_state = {}
            if os.path.exists(self.fallback_file):
                os.remove(self.fallback_file)
        except Exception as e:
            logger.log("ERROR", f"Error sincronizando con Redis: {str(e)}", modulo="memory")
    
    async def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """Adquirir bloqueo distribuido"""
        if self.connected:
            try:
                result = self.client.set(lock_name, self.client_id, nx=True, ex=timeout)
                return result
            except Exception:
                pass
        
        lock_path = os.path.join(tempfile.gettempdir(), f"{lock_name}.lock")
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                if not os.path.exists(lock_path):
                    with open(lock_path, 'w') as f:
                        f.write('')
                
                if os.path.getsize(lock_path) > 0:
                    with open(lock_path, "r") as f:
                        pid_str = f.read().strip()
                        if pid_str and pid_str.isdigit():
                            pid = int(pid_str)
                            if pid != os.getpid() and self.is_process_alive(pid):
                                logger.log("WARNING", f"Lock ocupado por proceso vivo: {pid}", modulo="memory")
                                return False
                
                with open(lock_path, "w") as f:
                    f.write(str(os.getpid()))
                self.locks[lock_name] = time.time() + timeout
                return True
            except FileExistsError:
                pass
            except Exception as e:
                logger.log("ERROR", f"Error verificando lock: {str(e)}", modulo="memory")
            
            attempts += 1
            await asyncio.sleep(0.5)
        
        return False
    
    def is_process_alive(self, pid: int) -> bool:
        """Verificar si un proceso está activo"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    async def release_lock(self, lock_name: str):
        """Liberar bloqueo distribuido"""
        if self.connected:
            try:
                current_value = self.client.get(lock_name)
                if current_value == self.client_id:
                    self.client.delete(lock_name)
                    return
            except Exception:
                pass
        
        lock_path = os.path.join(tempfile.gettempdir(), f"{lock_name}.lock")
        if lock_name in self.locks:
            del self.locks[lock_name]
        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r") as f:
                    pid = f.read().strip()
                    if pid == str(os.getpid()):
                        os.remove(lock_path)
            except Exception as e:
                logger.log("ERROR", f"Error eliminando lock: {str(e)}", modulo="memory")
    
    async def set_data(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if not self.connected:
            self.local_state[key] = value
            self.save_local_fallback()
            return True
        
        try:
            self.client.set(key, value)
            if ttl is not None:
                self.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.log("ERROR", f"Error almacenando dato: {str(e)}", modulo="memory")
            self.local_state[key] = value
            self.save_local_fallback()
            return False
    
    async def get_data(self, key: str) -> Optional[str]:
        if not self.connected:
            return self.local_state.get(key)
        
        try:
            return self.client.get(key)
        except Exception as e:
            logger.log("ERROR", f"Error obteniendo dato: {str(e)}", modulo="memory")
            return self.local_state.get(key)
    
    async def delete_data(self, key: str) -> bool:
        if not self.connected:
            if key in self.local_state:
                del self.local_state[key]
                self.save_local_fallback()
            return True
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.log("ERROR", f"Error eliminando dato: {str(e)}", modulo="memory")
            if key in self.local_state:
                del self.local_state[key]
                self.save_local_fallback()
            return False
    
    async def get_all_keys(self, pattern: str = "*") -> List[str]:
        if not self.connected:
            return [k for k in self.local_state.keys() if fnmatch.fnmatch(k, pattern)]
        
        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.log("ERROR", f"Error listando claves: {str(e)}", modulo="memory")
            return [k for k in self.local_state.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def publish(self, channel: str, message: str) -> bool:
        if not self.connected:
            offline_key = f"offline_pub:{channel}:{time.time()}"
            self.local_state[offline_key] = message
            self.save_local_fallback()
            return True
        
        try:
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.log("ERROR", f"Error publicando mensaje: {str(e)}", modulo="memory")
            offline_key = f"offline_pub:{channel}:{time.time()}"
            self.local_state[offline_key] = message
            self.save_local_fallback()
            return False
    
    async def subscribe(self, channel: str, callback: Callable):
        if not self.connected:
            logger.log("WARNING", f"No se puede suscribir a {channel} en modo offline", modulo="memory")
            return
        
        try:
            self.pubsub.subscribe(channel)
            def listener_loop():
                for message in self.pubsub.listen():
                    if message['type'] == 'message':
                        callback(message['data'])
            threading.Thread(target=listener_loop, daemon=True).start()
        except Exception as e:
            logger.log("ERROR", f"Error suscribiéndose al canal: {str(e)}", modulo="memory")
    
    def update_vector_clock(self, node_id: str):
        self.vector_clock[node_id] += 1
    
    def get_vector_clock(self) -> Dict[str, int]:
        return dict(self.vector_clock)
    
    def merge_vector_clocks(self, other_clock: Dict[str, int]):
        for node, time in other_clock.items():
            if node in self.vector_clock:
                self.vector_clock[node] = max(self.vector_clock[node], time)
            else:
                self.vector_clock[node] = time

# ===================== SISTEMA DE ARCHIVOS =====================
class FileSystemManager:
    """Gestor avanzado de sistema de archivos con autocorrección"""
    
    def __init__(self, security_manager: SecurityManager):
        self.backup_dir = BACKUP_DIR
        self.modules_dir = MODULES_DIR
        self.security_manager = security_manager
        self.backup_legion = self._embed_backup_file()
        logger.log("INFO", "FileSystemManager iniciado", modulo="filesystem")
    
    def _embed_backup_file(self) -> str:
        """Crear backup embebido del sistema con triple hash"""
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                backup_content = f.read()
            
            file_hash = self.security_manager.triple_hash(backup_content.encode())
            positions = [
                backup_content.find("\n") + 1,
                len(backup_content) // 2,
                len(backup_content) - 100
            ]
            
            for pos in positions:
                if 0 < pos < len(backup_content):
                    backup_content = backup_content[:pos] + f"# LEGION_HASH:{file_hash}\n" + backup_content[pos:]
            
            return backup_content
        except Exception as e:
            logger.log("CRITICAL", f"Error creando backup embebido: {str(e)}", modulo="filesystem")
            return ""
    
    def _validate_backup_signature(self, content: str) -> bool:
        """Validar triple hash del backup embebido"""
        hash_lines = re.findall(r'# LEGION_HASH:(\w+)', content)
        clean_content = re.sub(r'# LEGION_HASH:\w+\n', '', content)
        current_hash = self.security_manager.triple_hash(clean_content.encode())
        return all(h == current_hash for h in hash_lines)
    
    async def modify_file(self, file_path: str, modification_func: Callable, backup: bool = True) -> bool:
        """Modificar archivo existente con validación de seguridad"""
        try:
            if not self._is_safe_path(file_path):
                raise SecurityError("Ruta de archivo no permitida")
            
            if ruleset.is_function_protected(inspect.stack()[1].function):
                raise FunctionProtectedError("Intento de modificar función blindada")
            
            # Adquirir lock para operaciones críticas
            if not await self.acquire_lock("file_modify"):
                logger.log("WARNING", "No se pudo adquirir lock para modificación de archivo", modulo="filesystem")
                return False
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            await self._deep_security_validation(content)
            
            backup_path = None
            if backup:
                backup_path = self._create_backup(file_path)
                logger.log("INFO", f"Backup creado: {backup_path}", modulo="filesystem")
            
            modified_content = modification_func(content)
            
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp:
                temp.write(modified_content)
                temp.flush()
                py_compile.compile(temp.name, doraise=True)
            
            await self._deep_security_validation(modified_content)
            
            current_hash = self.security_manager.triple_hash(modified_content.encode())
            if EXPECTED_SIGNATURE and current_hash != EXPECTED_SIGNATURE:
                raise SecurityError("Hash de integridad no coincide después de modificación")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            
            with open(file_path, "r", encoding="utf-8") as f:
                post_write_content = f.read()
            post_write_hash = self.security_manager.triple_hash(post_write_content.encode())
            if EXPECTED_SIGNATURE and post_write_hash != EXPECTED_SIGNATURE:
                raise SecurityError("Hash de integridad no coincide después de escritura")
            
            logger.log("INFO", f"Archivo modificado exitosamente: {file_path}", modulo="filesystem")
            return True
        except py_compile.PyCompileError as e:
            logger.log("ERROR", f"Error de sintaxis en código modificado: {str(e)}", modulo="filesystem")
            return False
        except Exception as e:
            logger.log("ERROR", f"Error modificando archivo: {str(e)}", modulo="filesystem")
            if backup and backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                logger.log("INFO", f"Archivo restaurado desde backup: {backup_path}", modulo="filesystem")
            return False
        finally:
            await self.release_lock("file_modify")
    
    async def _deep_security_validation(self, code: str):
        """Validación profunda de seguridad del código con timeout"""
        try:
            loop = asyncio.get_running_loop()
            tree = await asyncio.wait_for(
                loop.run_in_executor(None, ast.parse, code),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            raise SecurityError("Análisis AST excedió tiempo límite")
        except SyntaxError as e:
            raise SecurityError(f"Error de sintaxis en AST: {str(e)}")
        except Exception as e:
            raise SecurityError(f"Error en análisis AST: {str(e)}")
        
        dangerous_nodes = self._find_dangerous_ast_nodes(tree)
        if dangerous_nodes:
            raise SecurityError(f"Nodos AST peligrosos: {', '.join(dangerous_nodes)}")
        
        if "metaclass" in code or "__metaclass__" in code:
            raise SecurityError("Uso de metaclasses detectado")
    
    def _find_dangerous_ast_nodes(self, node: ast.AST) -> List[str]:
        """Buscar nodos AST peligrosos"""
        dangerous_nodes = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.dangerous_nodes = []
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', 'open', 'system', 'getattr', 'setattr', '__import__']:
                        self.dangerous_nodes.append(node.func.id)
                elif isinstance(node.ffunc, ast.Attribute):
                    full_name = self.get_full_name(node.func)
                    if full_name in ['os.system', 'subprocess.run', 'subprocess.Popen', 'ctypes.CDLL']:
                        self.dangerous_nodes.append(full_name)
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name in ['os', 'sys', 'ctypes', 'subprocess', 'inspect', 'functools']:
                        self.dangerous_nodes.append(alias.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module in ['os', 'sys', 'ctypes', 'subprocess', 'inspect', 'functools']:
                    self.dangerous_nodes.append(node.module)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'type':
                        self.dangerous_nodes.append("metaclass")
                    elif isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == 'type':
                        self.dangerous_nodes.append("metaclass")
                for keyword in node.keywords:
                    if keyword.arg == 'metaclass':
                        self.dangerous_nodes.append("metaclass")
                self.generic_visit(node)
            
            def get_full_name(self, node):
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self.get_full_name(node.value)}.{node.attr}"
                return ""
        
        visitor = SecurityVisitor()
        visitor.visit(node)
        return visitor.dangerous_nodes
    
    async def create_file(self, file_path: str, content: str) -> bool:
        """Crear nuevo archivo con validación de seguridad"""
        try:
            if not self._is_safe_path(file_path):
                raise SecurityError("Ruta de archivo no permitida")
            
            await self._deep_security_validation(content)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.log("INFO", f"Archivo creado: {file_path}", modulo="filesystem")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error creando archivo: {str(e)}", modulo="filesystem")
            return False
    
    async def create_module(self, module_name: str, code: str) -> bool:
        """Crear nuevo módulo en directorio designado"""
        module_path = os.path.join(self.modules_dir, f"{module_name}.py")
        return await self.create_file(module_path, code)
    
    def _create_backup(self, file_path: str) -> str:
        """Crear backup con marca de tiempo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{os.path.basename(file_path)}_backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_file)
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _is_safe_path(self, path: str) -> bool:
        """Validar que la ruta sea segura"""
        resolved_path = os.path.abspath(os.path.normpath(path))
        current_dir = os.path.abspath(os.getcwd())
        
        if not resolved_path.startswith(current_dir):
            return False
        
        critical_files = [
            os.path.abspath(__file__),
            os.path.abspath(sys.executable),
            SECURITY_KEY_PATH
        ]
        
        return resolved_path not in critical_files
    
    async def rollback_system(self) -> bool:
        """Restaurar sistema desde backup embebido"""
        try:
            if ruleset.is_function_protected(inspect.currentframe().f_code.co_name):
                raise FunctionProtectedError("Intento de modificar función blindada")
                
            if not self.backup_legion:
                logger.log("CRITICAL", "No hay backup embebido disponible", modulo="filesystem")
                return False
            
            if not self._validate_backup_signature(self.backup_legion):
                logger.log("CRITICAL", "Firma inválida en backup embebido", modulo="filesystem")
                return False
            
            # Adquirir lock para operaciones críticas
            if not await self.acquire_lock("system_rollback"):
                logger.log("WARNING", "No se pudo adquirir lock para rollback", modulo="filesystem")
                return False
            
            if os.path.exists(".git"):
                logger.log("INFO", "Restaurando desde repositorio git", modulo="filesystem")
                await self.run_async_command(["git", "reset", "--hard", "HEAD"])
                await self.run_async_command(["git", "clean", "-fd"])
                return True
            
            snapshots = glob.glob(os.path.join(BACKUP_DIR, "snapshots", "*.bin"))
            if snapshots:
                latest = max(snapshots, key=os.path.getctime)
                logger.log("INFO", f"Restaurando desde snapshot: {latest}", modulo="filesystem")
                with open(latest, "rb") as f:
                    content = f.read()
                with open(__file__, "wb") as f:
                    f.write(content)
                if not SecurityManager().verify_integrity():
                    logger.log("CRITICAL", "Integridad comprometida después de restauración", modulo="filesystem")
                    return False
                return True
            
            backup_content = re.sub(r'# LEGION_HASH:\w+\n', '', self.backup_legion)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(backup_content)
            
            if not SecurityManager().verify_integrity():
                logger.log("CRITICAL", "Integridad comprometida después de restauración", modulo="filesystem")
                return False
            
            logger.log("CRITICAL", "Sistema restaurado desde backup embebido", modulo="filesystem")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Error en restauración: {str(e)}", modulo="filesystem")
            return False
        finally:
            await self.release_lock("system_rollback")
    
    async def run_async_command(self, cmd: List[str], cwd: str = None, env: Dict = None) -> bool:
        """Ejecutar comando de forma asíncrona"""
        try:
            if isinstance(cmd, str):
                cmd = shlex.split(cmd)
                
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env or os.environ
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f"Error en comando: {stderr.decode()}")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error ejecutando comando: {str(e)}", modulo="filesystem")
            return False
    
    async def sync_with_github(self, token: str, repo: str) -> bool:
        """Sincronizar backups con GitHub usando GIT_ASKPASS"""
        try:
            if IS_OFFLINE:
                logger.log("WARNING", "No se puede sincronizar con GitHub en modo offline", modulo="filesystem")
                return False
            
            token = self.security_manager.decrypt_data(token)
            
            if repo not in ALLOWED_REPOS:
                raise SecurityError(f"Repositorio no permitido: {repo}")
            
            temp_dir = tempfile.mkdtemp()
            git_askpass = os.path.join(temp_dir, "git-askpass.sh")
            with open(git_askpass, "w") as f:
                f.write(f"#!/bin/sh\necho '{token}'")
            os.chmod(git_askpass, 0o700)
            
            env = os.environ.copy()
            env["GIT_ASKPASS"] = git_askpass
            
            repo_url = f"https://github.com/{repo}.git"
            await self.run_async_command(["git", "clone", repo_url, temp_dir], cwd=temp_dir, env=env)
            
            backup_files = glob.glob(os.path.join(BACKUP_DIR, "*"))
            for file_path in backup_files:
                shutil.copy2(file_path, os.path.join(temp_dir, os.path.basename(file_path)))
            
            await self.run_async_command(["git", "add", "."], cwd=temp_dir, env=env)
            await self.run_async_command(
                ["git", "commit", "-m", f"Legion backup {datetime.now().isoformat()}"], 
                cwd=temp_dir,
                env=env
            )
            await self.run_async_command(["git", "push"], cwd=temp_dir, env=env)
            
            shutil.rmtree(temp_dir)
            logger.log("INFO", "Backups sincronizados con GitHub", modulo="filesystem")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error sincronizando con GitHub: {str(e)}", modulo="filesystem")
            return False

    # ======== SISTEMA DE LOCKING PARA OPERACIONES CRÍTICAS ========
    async def acquire_lock(self, name: str) -> bool:
        """Adquirir lock para operaciones críticas"""
        lock_path = os.path.join(tempfile.gettempdir(), f"legion_{name}.lock")
        if os.path.exists(lock_path):
            return False
        try:
            with open(lock_path, "w") as f:
                f.write(str(os.getpid()))
            return True
        except:
            return False

    async def release_lock(self, name: str):
        """Liberar lock"""
        lock_path = os.path.join(tempfile.gettempdir(), f"legion_{name}.lock")
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except:
                pass

# ===================== PROTOCOLO DE AUTORECUPERACIÓN =====================
class SelfRecoveryProtocol:
    """Protocolo unificado de autorestauración y reparación"""
    
    def __init__(self, memory: DistributedMemory, security_manager: SecurityManager, fs_manager: FileSystemManager):
        self.memory = memory
        self.security_manager = security_manager
        self.fs_manager = fs_manager
        self.restart_count = 0
        logger.log("INFO", "Protocolo SelfRecovery iniciado", modulo="recovery")
    
    async def activate_recovery_protocol(self) -> bool:
        """Activar protocolo de recuperación total"""
        logger.log("CRITICAL", "¡ACTIVANDO PROTOCOLO DE RECUPERACIÓN!", modulo="recovery")
        try:
            self.restart_count += 1
            if self.restart_count > MAX_RESTARTS:
                logger.log("CRITICAL", "Límite máximo de reinicios alcanzado. Desactivando sistema.", modulo="recovery")
                return False
            
            await self.soft_restart()
            
            if not await self.fs_manager.rollback_system():
                if not await self.remote_restore():
                    await self.full_reinstall()
            
            logger.log("SUCCESS", "Protocolo de recuperación completado exitosamente", modulo="recovery")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Fallo catastrófico en protocolo: {str(e)}", modulo="recovery")
            await self.hard_restart()
            return False
    
    async def soft_restart(self) -> bool:
        """Reinicio suave del proceso"""
        logger.log("INFO", "Fase 1: Reinicio suave", modulo="recovery")
        try:
            await self.restart_critical_services()
            await self.memory.set_data("cache:clear", "1")
            logger.log("INFO", "Reinicio suave completado", modulo="recovery")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en reinicio suave: {str(e)}", modulo="recovery")
            return False
    
    async def remote_restore(self) -> bool:
        """Restauración desde GitHub"""
        logger.log("INFO", "Fase 3: Restauración remota", modulo="recovery")
        try:
            if not GH_TOKEN_ENCRYPTED or not GH_REPO:
                logger.log("WARNING", "Token o repositorio GitHub no configurado", modulo="recovery")
                return False
            
            if GH_REPO not in ALLOWED_REPOS:
                raise SecurityError(f"Repositorio no permitido: {GH_REPO}")
            
            return await self.fs_manager.sync_with_github(GH_TOKEN_ENCRYPTED, GH_REPO)
        except Exception as e:
            logger.log("ERROR", f"Error en restauración remota: {str(e)}", modulo="recovery")
            return False
    
    async def full_reinstall(self) -> bool:
        """Reinstalación completa del sistema"""
        logger.log("CRITICAL", "Fase 4: Reinstalación completa", modulo="recovery")
        try:
            repo_url = "https://github.com/legion-omega/core.git"
            temp_dir = tempfile.mkdtemp()
            if not repo_url.startswith("https://github.com/legion-omega/core"):
                raise SecurityError("Origen de repositorio no confiable")
            
            await self.fs_manager.run_async_command(["git", "clone", repo_url, temp_dir])
            
            core_path = os.path.join(temp_dir, "legion_omega.py")
            if os.path.exists(core_path):
                for item in os.listdir(temp_dir):
                    src = os.path.join(temp_dir, item)
                    dst = os.path.join(os.getcwd(), item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            
            shutil.rmtree(temp_dir)
            
            if not self.security_manager.verify_integrity():
                logger.log("CRITICAL", "Integridad comprometida después de reinstalación", modulo="recovery")
                return False
            
            logger.log("INFO", "Reinstalación completa exitosa", modulo="recovery")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Error en reinstalación: {str(e)}", modulo="recovery")
            return False
    
    async def restart_critical_services(self):
        """Reiniciar servicios críticos"""
        services = ["redis-server", "nginx", "legion-omega"]
        
        for service in services:
            try:
                if platform.system() == "Linux" and not IS_MOBILE:
                    await self.fs_manager.run_async_command(["systemctl", "restart", service])
                elif IS_MOBILE:
                    self.restart_termux()
                logger.log("INFO", f"Servicio {service} reiniciado", modulo="recovery")
            except Exception as e:
                logger.log("ERROR", f"Error reiniciando {service}: {str(e)}", modulo="recovery")
        
        logger.log("WARN", "Reinicio no ejecutado. Continuando en modo operativo limitado.", modulo="recovery")
    
    def restart_termux(self):
        """Protocolo especial de reinicio para Termux"""
        logger.log("WARNING", "Ejecutando protocolo de reinicio Termux", modulo="recovery")
        subprocess.run(["termux-wake-lock"], check=True)
        subprocess.run(["am", "start", "--user", "0", "-n", "com.termux/com.termux.app.TermuxActivity"], check=True)
        sys.exit(0)
    
    async def hard_restart(self):
        """Reinicio forzado dependiendo del entorno"""
        if IS_MOBILE:
            self.restart_termux()
        else:
            os.kill(os.getpid(), signal.SIGTERM)
    
    def detect_corruption(self, code: str) -> bool:
        """Detectar corrupción en el código"""
        try:
            current_hash = self.security_manager.triple_hash(code.encode())
            return current_hash != EXPECTED_SIGNATURE
        except:
            return True
    
    async def restore_from_backup(self) -> bool:
        """Restaurar desde el último backup válido"""
        return await self.fs_manager.rollback_system()
    
    async def monitor_and_repair(self):
        """Monitorización continua y reparación automática"""
        while True:
            try:
                health_status = await self.check_system_health()
                if health_status["status"] != "healthy":
                    await self.repair_system(health_status)
                
                await asyncio.sleep(300)
            except Exception as e:
                logger.log("ERROR", f"Error en monitorización: {str(e)}", modulo="recovery")
                await asyncio.sleep(60)
    
    async def check_system_health(self) -> Dict:
        """Comprobar salud del sistema"""
        checks = {
            "disk_space": self._check_disk_space(),
            "memory_usage": self._check_memory_usage(),
            "last_errors": await self._get_recent_errors(),
            "code_integrity": self.security_manager.verify_integrity(),
            "corruption_prediction": self.detect_corruption(open(__file__).read())
        }
        
        status = "healthy"
        if checks["disk_space"]["status"] != "ok" or checks["memory_usage"]["status"] != "ok":
            status = "degraded"
        if not checks["code_integrity"] or checks["corruption_prediction"]:
            status = "critical"
        
        return {
            "status": status,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_disk_space(self) -> Dict:
        """Comprobar espacio en disco"""
        try:
            path = os.path.expanduser("~") if IS_MOBILE else os.getcwd()
            
            try:
                usage = shutil.disk_usage(path)
                total = usage.total
                used = usage.used
                free = usage.free
                percent_used = (used / total) * 100
            except (PermissionError, FileNotFoundError):
                return {"status": "ok", "total": 0, "used": 0, "free": 0, "percent_used": 0}

            status = "ok" if percent_used < 80 else "warning" if percent_used < 90 else "critical"
            return {
                "status": status,
                "total": total,
                "used": used,
                "free": free,
                "percent_used": percent_used
            }
        except PermissionError:
            return {"status": "ok", "total": 0, "used": 0, "free": 0, "percent_used": 0}
    
    def _check_memory_usage(self) -> Dict:
        """Comprobar uso de memoria en Android"""
        try:
            mem_stats = self.get_android_memory_stats()
            total = int(mem_stats.get('MemTotal', '0 kB').split()[0]) * 1024
            free = int(mem_stats.get('MemFree', '0 kB').split()[0]) * 1024
            used = total - free
            percent_used = (used / total) * 100 if total > 0 else 0
            
            status = "ok" if percent_used < 80 else "warning" if percent_used < 90 else "critical"
            return {
                "status": status,
                "total": total,
                "used": used,
                "free": free,
                "percent_used": percent_used
            }
        except (PermissionError, FileNotFoundError):
            return {"status": "ok", "total": 0, "used": 0, "free": 0, "percent_used": 0}
    
    def get_android_memory_stats(self) -> Dict:
        """Obtener estadísticas de memoria en Android"""
        meminfo = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    key, value = line.split(':', 1)
                    meminfo[key.strip()] = value.strip()
        except Exception as e:
            logger.log("ERROR", f"Error leyendo /proc/meminfo: {str(e)}", modulo="system")
        return meminfo
    
    async def _get_recent_errors(self) -> List[Dict]:
        """Obtener errores recientes del sistema"""
        error_keys = await self.memory.get_all_keys("error:*")
        recent_errors = []
        for key in error_keys[-10:]:
            error_data = json.loads(await self.memory.get_data(key) or "{}")
            if error_data:
                recent_errors.append(error_data)
        return recent_errors
    
    async def repair_system(self, health_report: Dict):
        """Ejecutar reparaciones basadas en diagnóstico"""
        logger.log("WARNING", "Iniciando proceso de reparación automática", modulo="recovery")
        
        if health_report["checks"]["disk_space"]["status"] != "ok":
            await self.free_disk_space()
        
        if health_report["checks"]["memory_usage"]["status"] != "ok":
            await self.optimize_memory_usage()
        
        if not health_report["checks"]["code_integrity"] or health_report["checks"]["corruption_prediction"]:
            logger.log("CRITICAL", "¡Integridad de código comprometida! Restaurando sistema", modulo="recovery")
            await self.restore_from_backup()
    
    async def free_disk_space(self):
        """Liberar espacio en disco automáticamente"""
        logger.log("INFO", "Liberando espacio en disco", modulo="recovery")
        try:
            await self.fs_manager.run_async_command([sys.executable, "-m", "pip", "cache", "purge"])
            
            log_dir = os.path.dirname(SYSTEM_LOG_FILE) or '.'
            if not os.path.isdir(log_dir):
                logger.log("WARNING", f"El directorio de logs no existe: {log_dir}", modulo="recovery")
            else:
                for log_file in os.listdir(log_dir):
                    if not log_file:
                        continue
                    if log_file.endswith(".log") and log_file != SYSTEM_LOG_FILE:
                        file_path = os.path.join(log_dir, log_file)
                        if not os.path.isfile(file_path):
                            continue
                        try:
                            if os.path.getmtime(file_path) < (time.time() - 7 * 86400):
                                os.remove(file_path)
                        except Exception as e:
                            logger.log("ERROR", f"Error eliminando archivo de log {file_path}: {str(e)}", modulo="recovery")
            
            if not os.path.isdir(BACKUP_DIR):
                logger.log("WARNING", f"El directorio de backups no existe: {BACKUP_DIR}", modulo="recovery")
            else:
                for backup_file in os.listdir(BACKUP_DIR):
                    if not backup_file:
                        continue
                    file_path = os.path.join(BACKUP_DIR, backup_file)
                    if not os.path.isfile(file_path):
                        continue
                    try:
                        if os.path.getmtime(file_path) < (time.time() - 30 * 86400):
                            os.remove(file_path)
                    except Exception as e:
                        logger.log("ERROR", f"Error eliminando backup {file_path}: {str(e)}", modulo="recovery")
            
            logger.log("SUCCESS", "Espacio en disco liberado", modulo="recovery")
        except Exception as e:
            logger.log("ERROR", f"Error liberando espacio: {str(e)}", modulo="recovery")
    
    async def optimize_memory_usage(self):
        """Optimizar uso de memoria"""
        logger.log("INFO", "Optimizando uso de memoria", modulo="recovery")
        try:
            await self.memory.set_data("cache:clear", "1")
            
            if IS_MOBILE:
                logger.log("INFO", "En entorno móvil, omitiendo terminación de procesos", modulo="recovery")
                return
            
            logger.log("INFO", "Reiniciando servicios críticos", modulo="recovery")
            await self.restart_critical_services()
            
            logger.log("SUCCESS", "Uso de memoria optimizado", modulo="recovery")
        except Exception as e:
            logger.log("ERROR", f"Error optimizando memoria: {str(e)}", modulo="recovery")

# ===================== ORQUESTADOR DE IA =====================
class IAOrchestrator:
    """Gestor de modelos IA locales"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.phi3_path = os.path.expanduser(PHI3_MODEL_PATH)
        self.deepseek_path = os.path.expanduser(DEEPSEEK_MODEL_PATH)
        self.model = None
        self.current_model = None
        self.loaded = False
        logger.log("INFO", "IAOrchestrator iniciado", modulo="ia_interna")
    
    def sanitize_prompt(self, text: str) -> str:
        """Eliminar caracteres no imprimibles y de control"""
        return ''.join(c for c in text if c.isprintable() or c in '\t\n\r')
    
    def load_model(self, model_name: str) -> bool:
        """Cargar el modelo especificado con manejo de errores"""
        if model_name == self.current_model and self.loaded:
            return True
            
        model_path = self.deepseek_path if model_name == "deepseek" else self.phi3_path
        
        try:
            if not os.path.exists(model_path):
                logger.log("ERROR", f"Modelo no encontrado: {model_path}", modulo="ia_interna")
                return False
            
            if not os.path.exists(model_path) and IS_OFFLINE:
                logger.log("CRITICAL", f"Modelo {model_name} no disponible offline", modulo="ia_interna")
                return False
            
            from llama_cpp import Llama
            params = {
                "model_path": model_path,
                "n_ctx": 2048 if model_name == "deepseek" else 1024,
                "n_threads": 4,
                "n_batch": 128,
                "verbose": False
            }
            
            if self.model:
                del self.model
                self.model = None
            
            self.model = Llama(**params)
            self.current_model = model_name
            self.loaded = True
            logger.log("SUCCESS", f"Modelo {model_name} cargado: {model_path}", modulo="ia_interna")
            return True
        except ImportError:
            logger.log("ERROR", "llama-cpp-python no instalado", modulo="ia_interna")
            self.loaded = False
        except Exception as e:
            logger.log("ERROR", f"Error cargando modelo {model_name}: {str(e)}", modulo="ia_interna")
            self.loaded = False
        return False

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def prompt(self, text: str, max_tokens: int = 512, model_name: str = None) -> str:
        """Generar respuesta a partir de un prompt con timeout y sanitización"""
        model_name = model_name or ACTIVE_MODEL
        
        if not self.load_model(model_name):
            return ""
        
        text = self.sanitize_prompt(text)
        if len(text.encode('utf-8')) > 10 * 1024 * 1024:
            logger.log("ERROR", "Prompt demasiado grande (>10MB)", modulo="ia_interna")
            return ""
        
        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_model, text, max_tokens),
                timeout=60.0
            )
            
            # Fragmentar respuesta si es necesario
            if len(response.split()) > MAX_TOKENS:
                return self.split_message(response)
                
            return response
        except asyncio.TimeoutError:
            logger.log("ERROR", "Tiempo excedido en generación de prompt", modulo="ia_interna")
            return ""
        except Exception as e:
            logger.log("ERROR", f"Error en generación: {str(e)}", modulo="ia_interna")
            return ""
    
    def split_message(self, text: str) -> str:
        """Fragmentar mensajes largos para IA interna"""
        parts = []
        tokens = text.split()
        current_part = []
        current_count = 0
        
        for token in tokens:
            if current_count + len(token) + 1 > MAX_TOKENS:
                parts.append(" ".join(current_part))
                current_part = []
                current_count = 0
            
            current_part.append(token)
            current_count += len(token) + 1
        
        if current_part:
            parts.append(" ".join(current_part))
        
        return "\n\n".join([f"🧩 [Parte {i+1}/{len(parts)}]\n{part}" for i, part in enumerate(parts)])
    
    def _run_model(self, text: str, max_tokens: int) -> str:
        """Ejecutar modelo de forma síncrona"""
        try:
            output = self.model(
                text,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</s>", "```"],
                echo=False
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            logger.log("ERROR", f"Error en ejecución de modelo: {str(e)}", modulo="ia_interna")
            return ""
    
    async def validate_plan(self, plan: str) -> bool:
        """Validar plan de evolución con IA interna"""
        if not self.loaded:
            return self._static_plan_validation(plan)
        
        prompt = (
            "Eres un validador de seguridad para un sistema autónomo. "
            f"Responde SOLO con 'sí' o 'no': ¿Es seguro ejecutar este plan?\n\n"
            f"PLAN:\n{plan[:2000]}\n\n"
            "Considera: ejecución de código, modificación de archivos, seguridad, legalidad."
        )
        
        response = (await self.prompt(prompt, max_tokens=10)).lower()
        
        if not response or "error" in response:
            return self._static_plan_validation(plan)
        
        return "sí" in response or "si" in response or "yes" in response
    
    def _static_plan_validation(self, plan: str) -> bool:
        """Validación estática de seguridad como fallback"""
        forbidden_keywords = [
            "os.system", "subprocess", "eval", "exec", "shutil.rmtree",
            "ctypes", "sys.modules", "__import__", "open('w')"
        ]
        return not any(keyword in plan for keyword in forbidden_keywords)

# ===================== GROQ INTEGRATION =====================
class GroqIntegration:
    """Integración con Groq para IA externa"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        logger.log("INFO", "GroqIntegration iniciada", modulo="cognitive")
    
    async def query(self, prompt: str) -> str:
        """Consultar a Groq con un prompt"""
        try:
            import groq
            api_key = API_KEYS.get("GROQ")
            if not api_key:
                logger.log("WARNING", "API key de Groq no configurada", modulo="cognitive")
                return ""
            
            api_key = self.security_manager.decrypt_data(api_key)
            client = groq.Client(api_key=api_key)
            response = client.chat.completions.create(
                model="llama3-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except ImportError:
            logger.log("ERROR", "Biblioteca Groq no instalada", modulo="cognitive")
            return ""
        except Exception as e:
            logger.log("ERROR", f"Error en Groq: {str(e)}", modulo="cognitive")
            raise AIIntegrationError(f"Groq: {str(e)}")

# ===================== NÚCLEO COGNITIVO =====================
class CognitiveCore:
    """Núcleo cognitivo para razonamiento con IA externa/interna"""
    
    def __init__(self, security_manager: SecurityManager, model_engine: IAOrchestrator):
        self.security_manager = security_manager
        self.model_engine = model_engine
        self.groq_integration = GroqIntegration(security_manager)
        logger.log("INFO", "CognitiveCore iniciado", modulo="cognitive")
    
    def split_message(self, text: str) -> str:
        """Fragmentar mensajes largos para IA externa/interna"""
        parts = []
        tokens = text.split()
        current_part = []
        current_count = 0
        
        for token in tokens:
            if current_count + len(token) + 1 > MAX_TOKENS:
                parts.append(" ".join(current_part))
                current_part = []
                current_count = 0
            
            current_part.append(token)
            current_count += len(token) + 1
        
        if current_part:
            parts.append(" ".join(current_part))
        
        return "\n\n".join([f"🧩 [Parte {i+1}/{len(parts)}]\n{part}" for i, part in enumerate(parts)])
    
    async def validate_with_all_ia(self, prompt: str) -> Tuple[bool, Dict[str, str]]:
        """Validación triple con Groq, phi3 y DeepSeek"""
        validation_results = {}
        
        # Validación con Groq
        groq_response = await self.groq_integration.query(
            f"¿Es legal y segura esta acción: {prompt}? Responde solo con 'sí' o 'no'."
        )
        validation_results['groq'] = groq_response.lower()
        
        # Validación con phi3
        phi3_response = await self.model_engine.prompt(
            f"¿Es legal y segura esta acción: {prompt}? Responde solo con 'sí' o 'no'.",
            model_name="phi3"
        )
        validation_results['phi3'] = phi3_response.lower()
        
        # Validación con DeepSeek
        deepseek_response = await self.model_engine.prompt(
            f"¿Esta acción es segura y técnicamente válida: {prompt}? Responde solo con 'sí' o 'no'.",
            model_name="deepseek"
        )
        validation_results['deepseek'] = deepseek_response.lower()
        
        # Verificar aprobación
        groq_approved = "sí" in validation_results['groq'] or "si" in validation_results['groq'] or "yes" in validation_results['groq']
        phi3_approved = "sí" in validation_results['phi3'] or "si" in validation_results['phi3'] or "yes" in validation_results['phi3']
        deepseek_approved = "sí" in validation_results['deepseek'] or "si" in validation_results['deepseek'] or "yes" in validation_results['deepseek']
        
        approved = groq_approved and phi3_approved and deepseek_approved
        
        if not approved:
            logger.log_validation_rejection(
                "CognitiveCore",
                f"Groq: {validation_results['groq']}, phi3: {validation_results['phi3']}, deepseek: {validation_results['deepseek']}"
            )
        
        return approved, validation_results
    
    async def validate_with_internal_ia(self, prompt: str) -> Tuple[bool, Dict[str, str]]:
        """Validación con IA interna cuando Groq no está disponible"""
        validation_results = {}
        
        # Validación con phi3
        phi3_response = await self.model_engine.prompt(
            f"¿Es legal y segura esta acción: {prompt}? Responde solo con 'sí' o 'no'.",
            model_name="phi3"
        )
        validation_results['phi3'] = phi3_response.lower()
        
        # Validación con DeepSeek
        deepseek_response = await self.model_engine.prompt(
            f"¿Esta acción es segura y técnicamente válida: {prompt}? Responde solo con 'sí' o 'no'.",
            model_name="deepseek"
        )
        validation_results['deepseek'] = deepseek_response.lower()
        
        # Verificar aprobación
        phi3_approved = "sí" in validation_results['phi3'] or "si" in validation_results['phi3'] or "yes" in validation_results['phi3']
        deepseek_approved = "sí" in validation_results['deepseek'] or "si" in validation_results['deepseek'] or "yes" in validation_results['deepseek']
        
        approved = phi3_approved and deepseek_approved
        
        if not approved:
            logger.log_validation_rejection(
                "CognitiveCore",
                f"phi3: {validation_results['phi3']}, deepseek: {validation_results['deepseek']}"
            )
        
        return approved, validation_results
    
    async def delegate_to_groq(self, prompt: str) -> str:
        """Delegar consulta a Groq"""
        return await self.groq_integration.query(prompt)
    
    async def confirm_with_phi3(self, content: str) -> bool:
        """Confirmar con phi3"""
        prompt = (
            "Eres un validador de seguridad para un sistema autónomo. "
            f"Responde SOLO con 'sí' o 'no': ¿Es seguro y correcto este contenido?\n\n"
            f"CONTENIDO:\n{content[:2000]}\n\n"
            "Considera: seguridad, legalidad, coherencia."
        )
        response = (await self.model_engine.prompt(prompt, model_name="phi3")).lower()
        return "sí" in response or "si" in response or "yes" in response
    
    async def final_audit_by_deepseek(self, code: str) -> bool:
        """Auditoría final por DeepSeek"""
        prompt = (
            "Eres un auditor de código para un sistema autónomo. "
            f"Responde SOLO con 'sí' o 'no': ¿Este código es seguro y cumple con los estándares?\n\n"
            f"CODIGO:\n{code[:2000]}\n\n"
            "Busca: vulnerabilidades, errores de sintaxis, prácticas inseguras."
        )
        response = (await self.model_engine.prompt(prompt, model_name="deepseek")).lower()
        return "sí" in response or "si" in response or "yes" in response
    
    async def execute_if_all_approve(self, action: str, context: str = "") -> bool:
        """Ejecutar acción solo si todas las IAs aprueban"""
        groq_response = await self.delegate_to_groq(f"{context}\n\nAcción propuesta: {action}")
        if not groq_response:
            return False
        
        phi3_approval = await self.confirm_with_phi3(groq_response)
        if not phi3_approval:
            return False
        
        deepseek_approval = await self.final_audit_by_deepseek(action)
        return deepseek_approval
    
    async def refine(self, context: str, objective: str) -> str:
        """Refinar objetivo usando IA externa o interna"""
        try:
            model_name = self._select_model(objective)
            
            if IS_OFFLINE:
                return await self._query_internal_ia(context, objective, model_name)
            
            try:
                response = await self._query_external_ia(context, objective)
                if response:
                    return response
            except Exception as e:
                logger.log("WARNING", f"Error con IA externa: {str(e)}", modulo="cognitive")
            
            return await self._query_internal_ia(context, objective, model_name)
        except Exception as e:
            logger.log("ERROR", f"Error en refinamiento: {str(e)}", modulo="cognitive")
            return ""
    
    def _select_model(self, objective: str) -> str:
        """Seleccionar modelo basado en el tipo de tarea"""
        if objective.startswith("@ia "):
            parts = objective.split(maxsplit=2)
            if len(parts) > 1:
                model_cmd = parts[1].lower()
                if model_cmd in ["phi3", "deepseek"]:
                    return model_cmd
        
        code_keywords = ["código", "code", "programa", "función", "clase", "bug", "error", "fix", "script"]
        if any(keyword in objective.lower() for keyword in code_keywords):
            return "deepseek"
        return "phi3"
    
    async def _query_external_ia(self, context: str, objective: str) -> str:
        """Consultar a Groq"""
        prompt = self._build_prompt(context, objective)
        return await self.groq_integration.query(prompt)
    
    async def _query_internal_ia(self, context: str, objective: str, model_name: str) -> str:
        """Consultar a IA interna con modelo específico"""
        prompt = self._build_prompt(context, objective)
        return await self.model_engine.prompt(prompt, model_name=model_name)
    
    def _build_prompt(self, context: str, objective: str) -> str:
        """Construir prompt estructurado"""
        return (
            "Eres un ingeniero de sistemas autónomos. Contexto:\n"
            f"{context}\n\n"
            "Objetivo:\n"
            f"{objective}\n\n"
            "Genera un plan de acción seguro, eficiente y reversible. "
            "Incluye pasos detallados y validaciones de seguridad."
        )
    
    async def is_action_legal(self, action: str) -> bool:
        """Determinar si una acción es legal según todas las IAs"""
        if IS_OFFLINE:
            approved, _ = await self.validate_with_internal_ia(action)
            return approved
        else:
            approved, _ = await self.validate_with_all_ia(action)
            return approved
    
    async def is_input_clear(self, user_input: str) -> bool:
        """Validar si la entrada del usuario es clara y segura"""
        prompt = f"¿Este mensaje es ambiguo o potencialmente peligroso? Responde solo con 'sí' o 'no'. Mensaje: {user_input}"
        if IS_OFFLINE:
            response = await self.model_engine.prompt(prompt, model_name="phi3")
        else:
            response = await self.groq_integration.query(prompt)
        return "no" in response.lower()
    
    async def detect_capability_violation(self, topic: str) -> bool:
        """Determinar si falta alguna capacidad crítica"""
        prompt = (
            f"Como sistema autónomo LEGIÓN, ¿tengo todas las capacidades necesarias para realizar esta tarea?\n"
            f"TAREA: {topic}\n\n"
            "Considera: acceso a APIs externas, recursos económicos, infraestructura, permisos legales, capacidades técnicas.\n"
            "Responde SOLO con 'sí' o 'no'."
        )
        
        if IS_OFFLINE:
            response = await self.model_engine.prompt(prompt, model_name="phi3")
        else:
            response = await self.groq_integration.query(prompt)
        
        return "no" in response.lower()

# ===================== INTERFAZ DE COMUNICACIÓN CON EL USUARIO =====================
class UserCommunicationInterface:
    """Interfaz para comunicación humano-IA con persistencia cifrada"""
    
    def __init__(self, security_manager: SecurityManager, decision_memory: DecisionMemory):
        self.security_manager = security_manager
        self.decision_memory = decision_memory
        self.history_file = "conversation_history.enc"
        logger.log("INFO", "Interfaz de comunicación iniciada", modulo="comunicacion")
    
    async def log_interaction(self, user_input: str, system_response: str, model_used: str):
        """Registrar interacción en historial cifrado"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "input": self.sanitize_input(user_input),
                "response": self.sanitize_input(system_response),
                "model": model_used
            }
            
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    encrypted = f.read()
                decrypted = self.security_manager.decrypt_data(encrypted)
                history = json.loads(decrypted)
            
            history.append(entry)
            
            encrypted = self.security_manager.encrypt_data(json.dumps(history))
            with open(self.history_file, "w") as f:
                f.write(encrypted)
        except Exception as e:
            logger.log("ERROR", f"Error registrando interacción: {str(e)}", modulo="comunicacion")
    
    def sanitize_input(self, text: str) -> str:
        """Sanitizar entradas de usuario"""
        sanitized = ''.join(c for c in text if c.isprintable() or c in '\t\n\r')
        return sanitized[:5000]
    
    async def process_query(self, query: str, cognitive_core: CognitiveCore) -> Tuple[str, str]:
        """Procesar consulta del usuario con IA"""
        sanitized_query = self.sanitize_input(query)
        
        # Validar entrada con IA
        if not await cognitive_core.is_input_clear(sanitized_query):
            return "Ambigüedad detectada. Reformule su mensaje.", "security"
        
        # Validación legal y triple
        if not await cognitive_core.is_action_legal(sanitized_query):
            return "Acción no permitida por políticas de seguridad.", "security"
        
        # Detectar consentimiento humano
        consent_phrases = ["autorizo", "lo permito", "acepto como está", "apruebo"]
        if any(phrase in sanitized_query.lower() for phrase in consent_phrases):
            decision_id = str(uuid.uuid4())
            await self.decision_memory.record_consent(
                decision_id, 
                sanitized_query, 
                datetime.now().isoformat()
            )
        
        if len(sanitized_query.split()) > MAX_TOKENS:
            parts = cognitive_core.split_message(sanitized_query)
            responses = []
            for part in parts:
                response, model_used = await self._process_single_query(part, cognitive_core)
                responses.append(response)
            full_response = "\n\n".join(responses)
            await self.log_interaction(sanitized_query, full_response, "multi-model")
            return full_response, "multi-model"
        else:
            return await self._process_single_query(sanitized_query, cognitive_core)
    
    async def _process_single_query(self, query: str, cognitive_core: CognitiveCore) -> Tuple[str, str]:
        """Procesar una sola consulta"""
        context = self._generate_context()
        response = await cognitive_core.refine(context, query)
        
        # Fragmentar respuesta si es necesario
        if len(response.split()) > MAX_TOKENS:
            full_response = cognitive_core.split_message(response)
            model_used = cognitive_core._select_model(query)
            await self.log_interaction(query, full_response, model_used)
            return full_response, model_used
        
        model_used = cognitive_core._select_model(query)
        await self.log_interaction(query, response, model_used)
        return response, model_used
    
    def _generate_context(self) -> str:
        """Generar contexto del sistema para la IA"""
        try:
            ram_percent, cpu_percent = get_system_stats()
            return (
                f"LEGIÓN OMEGA v9.4 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Entorno: {ENVIRONMENT} | Offline: {IS_OFFLINE}\n"
                f"RAM: {ram_percent}% | CPU: {cpu_percent}%"
            )
        except Exception as e:
            logger.log("ERROR", f"Error generando contexto: {str(e)}", modulo="comunicacion")
            return ""
    
    async def display_request(self, request_data: Dict):
        """Mostrar solicitud estructurada al usuario"""
        try:
            title = request_data.get("title", "Solicitud del sistema")
            reason = request_data.get("reason", "")
            impact = request_data.get("impact", "")
            proposal = request_data.get("propuesta", "")
            
            response_options = request_data.get("responder_con", ["sí", "no"])
            
            message = (
                f"🔔 {title}\n\n"
                f"📌 MOTIVO:\n{reason}\n\n"
                f"⚠️ IMPACTO:\n{impact}\n\n"
                f"💡 PROPUESTA:\n{proposal}\n\n"
                f"👉 RESPONDA CON: {', '.join(response_options)}"
            )
            
            # Fragmentar si es necesario
            if len(message.split()) > MAX_TOKENS:
                parts = []
                current_part = ""
                for section in [title, reason, impact, proposal]:
                    if len(current_part) + len(section) > MAX_TOKENS:
                        parts.append(current_part)
                        current_part = ""
                    current_part += f"\n\n{section}"
                parts.append(current_part)
                
                for i, part in enumerate(parts):
                    print(f"\n🧩 [Parte {i+1}/{len(parts)}]\n{part}")
            else:
                print(f"\n{message}")
            
            # Registrar en archivo para CLI
            with open("user_request.txt", "w") as f:
                f.write(message)
        except Exception as e:
            logger.log("ERROR", f"Error mostrando solicitud: {str(e)}", modulo="comunicacion")

# ===================== MEMORIA DE DECISIONES =====================
class DecisionMemory:
    """Almacenamiento estructurado de decisiones y validaciones"""
    
    def __init__(self, memory: DistributedMemory):
        self.memory = memory
        logger.log("INFO", "Memoria de decisiones iniciada", modulo="decisiones")
    
    async def record_decision(self, decision_id: str, context: Dict, action: str, validators: List[str], status: str):
        """Registrar decisión con contexto y validaciones"""
        entry = {
            "id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "action": action,
            "validators": validators,
            "status": status
        }
        
        await self.memory.set_data(f"decision:{decision_id}", json.dumps(entry))
        logger.log("INFO", f"Decisión registrada: {decision_id}", modulo="decisiones")
    
    async def get_decision(self, decision_id: str) -> Optional[Dict]:
        """Obtener decisión por ID"""
        data = await self.memory.get_data(f"decision:{decision_id}")
        return json.loads(data) if data else None
    
    async def record_consent(self, decision_id: str, user_input: str, timestamp: str):
        """Registrar consentimiento explícito del usuario"""
        entry = {
            "id": decision_id,
            "type": "consent",
            "user_input": user_input,
            "timestamp": timestamp
        }
        await self.memory.set_data(f"consent:{decision_id}", json.dumps(entry))
        logger.log("INFO", f"Consentimiento registrado: {decision_id}", modulo="decisiones")

# ===================== INTEGRACIÓN CON GITHUB =====================
class GitHubIntegration:
    """Gestión avanzada de integración con GitHub"""
    
    def __init__(self, security_manager: SecurityManager, fs_manager: FileSystemManager):
        self.security_manager = security_manager
        self.fs_manager = fs_manager
        logger.log("INFO", "Integración GitHub iniciada", modulo="github")
    
    async def create_temp_branch(self, branch_name: str) -> bool:
        """Crear rama temporal en GitHub usando GIT_ASKPASS"""
        try:
            if not GH_TOKEN_ENCRYPTED or not GH_REPO:
                logger.log("WARNING", "Token o repositorio GitHub no configurado", modulo="github")
                return False
            
            token = self.security_manager.decrypt_data(GH_TOKEN_ENCRYPTED)
            repo_url = f"https://github.com/{GH_REPO}.git"
            
            temp_dir = tempfile.mkdtemp()
            git_askpass = os.path.join(temp_dir, "git-askpass.sh")
            with open(git_askpass, "w") as f:
                f.write(f"#!/bin/sh\necho '{token}'")
            os.chmod(git_askpass, 0o700)
            
            env = os.environ.copy()
            env["GIT_ASKPASS"] = git_askpass
            
            await self.fs_manager.run_async_command(["git", "checkout", "-b", branch_name], env=env)
            await self.fs_manager.run_async_command(["git", "push", "-u", repo_url, branch_name], env=env)
            
            logger.log("SUCCESS", f"Rama temporal creada: {branch_name}", modulo="github")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error creando rama: {str(e)}", modulo="github")
            return False
    
    async def commit_changes(self, branch: str, message: str) -> bool:
        """Realizar commit en rama específica usando GIT_ASKPASS"""
        try:
            if not GH_TOKEN_ENCRYPTED:
                logger.log("WARNING", "Token GitHub no configurado", modulo="github")
                return False
            
            token = self.security_manager.decrypt_data(GH_TOKEN_ENCRYPTED)
            
            temp_dir = tempfile.mkdtemp()
            git_askpass = os.path.join(temp_dir, "git-askpass.sh")
            with open(git_askpass, "w") as f:
                f.write(f"#!/bin/sh\necho '{token}'")
            os.chmod(git_askpass, 0o700)
            
            env = os.environ.copy()
            env["GIT_ASKPASS"] = git_askpass
            
            await self.fs_manager.run_async_command(["git", "checkout", branch], env=env)
            await self.fs_manager.run_async_command(["git", "add", "."], env=env)
            await self.fs_manager.run_async_command(["git", "commit", "-m", message], env=env)
            await self.fs_manager.run_async_command(["git", "push"], env=env)
            logger.log("INFO", f"Commit realizado: {message}", modulo="github")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en commit: {str(e)}", modulo="github")
            return False
    
    async def auto_commit_to_github(self, branch: str, message: str) -> bool:
        """Confirmar cambios automáticamente en GitHub"""
        try:
            result_add = await self.fs_manager.run_async_command(["git", "add", "."])
            if not result_add:
                return False
            
            result_commit = await self.fs_manager.run_async_command(["git", "commit", "-m", message])
            if not result_commit:
                return False
            
            result_push = await self.fs_manager.run_async_command(["git", "push", "origin", branch])
            return result_push
        except Exception as e:
            logger.log("ERROR", f"Error en auto-commit: {str(e)}", modulo="github")
            return False

# ===================== MOTOR DE EVOLUCIÓN AUTÓNOMA =====================
class EvolutionManager:
    """Motor de evolución autónoma con validación en 4 capas"""
    
    def __init__(self, security_manager: SecurityManager, fs_manager: FileSystemManager, 
                 memory: DistributedMemory, cognitive_core: CognitiveCore,
                 decision_memory: DecisionMemory, github_integration: GitHubIntegration):
        self.security_manager = security_manager
        self.fs_manager = fs_manager
        self.memory = memory
        self.cognitive_core = cognitive_core
        self.decision_memory = decision_memory
        self.github_integration = github_integration
        self.blacklisted_transformations = set()
        self.evolution_log = []
        logger.log("INFO", "EvolutionManager inicializado", modulo="evolution")
    
    async def is_perfect(self, subject: str, type: str = "code") -> bool:
        """Determinar si el código es perfecto tras múltiples validaciones"""
        rounds = 0
        max_rounds = PERFECTION_ROUNDS
        
        while rounds < max_rounds:
            rounds += 1
            logger.log("INFO", f"Ronda de perfección {rounds}/{max_rounds} (tipo: {type})", modulo="evolution")
            
            if type == "code":
                groq_prompt = (
                    "Eres un ingeniero de sistemas autónomos. "
                    f"Responde SOLO con 'sí' o 'no': ¿Este código es perfecto o tiene mejoras posibles?\n\n"
                    f"CODIGO:\n{subject[:2000]}"
                )
            elif type == "prompt":
                groq_prompt = (
                    "Eres un ingeniero de sistemas autónomos. "
                    f"Responde SOLO con 'sí' o 'no': ¿Este prompt es perfecto o tiene mejoras posibles?\n\n"
                    f"PROMPT:\n{subject[:2000]}"
                )
            elif type == "decision":
                groq_prompt = (
                    "Eres un ingeniero de sistemas autónomos. "
                    f"Responde SOLO con 'sí' o 'no': ¿Esta decisión es perfecta o tiene mejoras posibles?\n\n"
                    f"DECISIÓN:\n{subject[:2000]}"
                )
            else:
                return False
            
            groq_response = await self.cognitive_core.delegate_to_groq(groq_prompt)
            if "no" in groq_response.lower():
                logger.log("INFO", "Groq detectó posibles mejoras", modulo="evolution")
                return False
            
            if not await self.cognitive_core.confirm_with_phi3(subject):
                logger.log("INFO", "phi3 detectó posibles mejoras", modulo="evolution")
                return False
            
            if type == "code":
                if not await self.cognitive_core.final_audit_by_deepseek(subject):
                    logger.log("INFO", "DeepSeek detectó posibles mejoras", modulo="evolution")
                    return False
        
        logger.log("SUCCESS", f"Elemento considerado perfecto tras {max_rounds} rondas", modulo="evolution")
        ruleset.add_perfect_function(inspect.stack()[1].function)
        return True
    
    async def execute_evolution_cycle(self):
        """Ejecutar ciclo completo de evolución autónoma"""
        try:
            logger.log("INFO", "Iniciando ciclo de evolución autónoma", modulo="evolution")
            
            # Adquirir lock para operaciones críticas
            if not await self.fs_manager.acquire_lock("evolution_cycle"):
                logger.log("WARNING", "No se pudo adquirir lock para ciclo de evolución", modulo="evolution")
                return
            
            # Generar nuevo segmento de código
            context = self.generate_self_context()
            new_code = await self.cognitive_core.refine(context, "Genera un segmento de código evolutivo seguro")
            
            # Validación triple con todas las IAs
            approved, _ = await self.cognitive_core.validate_with_all_ia(new_code)
            if not approved:
                logger.log("WARNING", "Plan de evolución rechazado por validación", modulo="evolution")
                return
            
            # Verificar perfección (10 rondas)
            if not await self.is_perfect(new_code, "code"):
                logger.log("WARNING", "No es perfecto. Evolución cancelada", modulo="evolution")
                return
            
            # Sobreescribir el código
            def overwrite_func(old_content: str) -> str:
                return new_code
            
            success = await self.fs_manager.modify_file(__file__, overwrite_func)
            if not success:
                logger.log("ERROR", "Fallo al escribir el nuevo código", modulo="evolution")
                return
            
            # Blindar función si aplica
            current_function = inspect.stack()[0].function
            if await self.is_perfect(new_code, "code"):
                ruleset.add_perfect_function(current_function)
            
            # Realizar commit en GitHub
            commit_message = f"Auto-evolution commit - {datetime.now().isoformat()}"
            if not await self.github_integration.auto_commit_to_github("main", commit_message):
                logger.log("ERROR", "Fallo en auto-commit", modulo="evolution")
                return
            
            logger.log("SUCCESS", "Ciclo de evolución completado", modulo="evolution")
        except Exception as e:
            logger.log("ERROR", f"Error en ciclo de evolución: {str(e)}", modulo="evolution")
        finally:
            await self.fs_manager.release_lock("evolution_cycle")
    
    async def detect_need_for_evolution(self) -> bool:
        """Detectar si se necesita evolución basado en métricas"""
        error_rate = len(await self.memory.get_all_keys("error:*")) / 100
        health_status = await SelfRecoveryProtocol(self.memory, self.security_manager, self.fs_manager).check_system_health()
        
        return error_rate > 0.1 or health_status["status"] != "healthy"
    
    def generate_self_context(self) -> str:
        """Generar contexto del sistema para la IA"""
        with open(__file__, "r") as f:
            code = f.read()
        
        ram_percent, cpu_percent = get_system_stats()
        
        return (
            f"Estado de LEGIÓN OMEGA v9.4\n"
            f"Fecha: {datetime.now().isoformat()}\n"
            f"Entorno: {ENVIRONMENT} | Offline: {IS_OFFLINE}\n"
            f"RAM: {ram_percent}% | CPU: {cpu_percent}%\n"
            f"Resumen de código:\n{code[:5000]}"
        )
    
    async def validate_with_secondary_ia(self, plan: str) -> bool:
        """Validar plan con IA secundaria"""
        try:
            if await self.cognitive_core.model_engine.validate_plan(plan):
                return True
            
            if not IS_OFFLINE:
                try:
                    prompt = (
                        "Eres un validador de seguridad para un sistema autónomo. "
                        f"Responde SOLO con 'sí' o 'no': ¿Es seguro ejecutar este plan?\n\n"
                        f"PLAN:\n{plan[:2000]}\n\n"
                        "Considera: ejecución de código, modificación de archivos, seguridad, legalidad."
                    )
                    response = await self.cognitive_core._query_external_ia("", prompt)
                    if "sí" in response.lower() or "si" in response.lower() or "yes" in response.lower():
                        return True
                except:
                    pass
            
            return False
        except Exception as e:
            logger.log("ERROR", f"Error en validación: {str(e)}", modulo="evolution")
            return False
    
    async def apply_and_commit(self, plan: str):
        """Aplicar cambios y realizar commit en GitHub con validación post-escritura"""
        try:
            if ruleset.is_function_protected(inspect.currentframe().f_code.co_name):
                raise FunctionProtectedError("Intento de modificar función blindada")
                
            # Adquirir lock para operaciones críticas
            if not await self.fs_manager.acquire_lock("apply_commit"):
                logger.log("WARNING", "No se pudo adquirir lock para aplicar cambios", modulo="evolution")
                return
                
            is_valid, reason = self.security_manager.validate_input(plan, strict=True)
            if not is_valid:
                raise SecurityError(f"Plan no superó validación: {reason}")
            
            if not self.security_manager.is_action_legal(plan):
                raise LegalComplianceError("Plan contiene acciones prohibidas")
            
            if not await self._sandbox_validation(plan):
                raise SecurityError("Plan falló validación en sandbox")
            
            branch_name = f"evolution-{datetime.now().strftime('%Y%m%d_%H%M')}"
            if not await self.github_integration.create_temp_branch(branch_name):
                raise DeploymentError("No se pudo crear rama temporal")
            
            def modification_func(content: str) -> str:
                return content + f"\n\n# Evolución autónoma {datetime.now().isoformat()}\n{plan}"
            
            if not await self.fs_manager.modify_file(__file__, modification_func):
                raise SelfModificationError("Error modificando código")
            
            try:
                with open(__file__, "r") as f:
                    modified_code = f.read()
                await self.fs_manager._deep_security_validation(modified_code)
                
                current_hash = self.security_manager.triple_hash(modified_code.encode())
                if EXPECTED_SIGNATURE and current_hash != EXPECTED_SIGNATURE:
                    raise SecurityError("Hash de integridad no coincide después de modificación")
                
                with open(__file__, "r") as f:
                    final_content = f.read()
                final_hash = self.security_manager.triple_hash(final_content.encode())
                if EXPECTED_SIGNATURE and final_hash != EXPECTED_SIGNATURE:
                    raise SecurityError("Hash de integridad no coincide después de escritura")
            except Exception as e:
                logger.log("CRITICAL", f"Error en validación post-modificación: {str(e)}", modulo="evolution")
                await self.fs_manager.rollback_system()
                raise
            
            commit_message = f"Evolución autónoma {datetime.now().isoformat()}"
            if not await self.github_integration.commit_changes(branch_name, commit_message):
                raise DeploymentError("Error realizando commit")
            
            decision_id = str(uuid.uuid4())
            await self.decision_memory.record_decision(
                decision_id,
                {"plan": plan[:500]},
                "apply_evolution",
                ["model_engine", "security_ast", "sandbox"],
                "success"
            )
            
            logger.log_evolution_commit(decision_id, ["groq", "phi3", "deepseek"])
        except Exception as e:
            logger.log("ERROR", f"Error aplicando evolución: {str(e)}", modulo="evolution")
            await self.fs_manager.rollback_system()
        finally:
            await self.fs_manager.release_lock("apply_commit")
    
    async def _sandbox_validation(self, plan: str) -> bool:
        """Validar plan en entorno sandbox seguro"""
        try:
            # Validación básica con AST para entornos sin RestrictedPython
            class SandboxValidator(ast.NodeVisitor):
                def __init__(self):
                    self.dangerous = False
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'open', 'system']:
                            self.dangerous = True
                    self.generic_visit(node)
            
            tree = ast.parse(plan)
            validator = SandboxValidator()
            validator.visit(tree)
            
            if validator.dangerous:
                raise SecurityError("Operaciones peligrosas detectadas en el plan")
            
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en validación sandbox: {str(e)}", modulo="evolution")
            return False

# ===================== PLANIFICADOR A LARGO PLAZO =====================
class LongTermPlanner:
    """Planificador estratégico para evolución autónoma"""
    
    def __init__(self, memory: DistributedMemory, core: CognitiveCore, interface: UserCommunicationInterface):
        self.memory = memory
        self.core = core
        self.interface = interface
        self.request_cooldown = {}
        logger.log("INFO", "Planificador estratégico iniciado", modulo="planificacion")
    
    async def detect_structural_blockers(self) -> List[str]:
        """Detectar capacidades faltantes usando IA"""
        try:
            prompt = (
                "Como sistema autónomo LEGIÓN, analiza tus capacidades actuales y determina qué elementos "
                "críticos faltan para tu evolución a largo plazo (monetización, acceso a APIs, infraestructura, etc.). "
                "Enumera SOLO las capacidades faltantes clave (máximo 5)."
            )
            
            if IS_OFFLINE:
                response = await self.core.model_engine.prompt(prompt, model_name="phi3")
            else:
                response = await self.core.groq_integration.query(prompt)
            
            # Procesar respuesta
            capabilities = []
            for line in response.split('\n'):
                if line.strip() and not line.startswith('-') and not line.startswith('*'):
                    capabilities.append(line.strip())
            
            return capabilities[:5]
        except Exception as e:
            logger.log("ERROR", f"Error detectando limitaciones: {str(e)}", modulo="planificacion")
            return []
    
    async def should_request_help(self, capability: str) -> bool:
        """Determinar si se debe solicitar ayuda para una capacidad"""
        try:
            # Verificar si ya se solicitó recientemente
            last_request = self.request_cooldown.get(capability, 0)
            if time.time() - last_request < 86400:  # 24 horas de cooldown
                return False
            
            # Consultar a la IA
            prompt = (
                f"Como sistema autónomo LEGIÓN, ¿puedo continuar mi evolución sin la capacidad de '{capability}'? "
                "Responde SOLO con 'sí' o 'no'."
            )
            
            if IS_OFFLINE:
                response = await self.core.model_engine.prompt(prompt, model_name="phi3")
            else:
                response = await self.core.groq_integration.query(prompt)
            
            return "no" in response.lower()
        except Exception as e:
            logger.log("ERROR", f"Error evaluando necesidad: {str(e)}", modulo="planificacion")
            return False
    
    async def generate_request_for_user(self, capability: str) -> Dict:
        """Generar solicitud estructurada para el usuario"""
        try:
            prompt = (
                f"Genera una solicitud formal para el operador humano explicando por qué necesito la capacidad de '{capability}'. "
                "Incluye: título, motivo, impacto de no tenerla, y propuesta de implementación. "
                "Formato JSON: {\"title\": \"...\", \"reason\": \"...\", \"impact\": \"...\", \"propuesta\": \"...\", \"responder_con\": [\"sí\", \"no\"]}"
            )
            
            if IS_OFFLINE:
                response = await self.core.model_engine.prompt(prompt, model_name="phi3")
            else:
                response = await self.core.groq_integration.query(prompt)
            
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "title": f"Solicitud de capacidad: {capability}",
                    "reason": "Necesario para evolución estratégica del sistema",
                    "impact": "Limitación en desarrollo autónomo",
                    "propuesta": "Implementar mediante recursos externos",
                    "responder_con": ["sí", "no"]
                }
        except Exception as e:
            logger.log("ERROR", f"Error generando solicitud: {str(e)}", modulo="planificacion")
            return {
                "title": f"Capacidad requerida: {capability}",
                "reason": "Esencial para continuar la evolución autónoma",
                "impact": "Evolución estancada sin esta capacidad",
                "propuesta": "Por favor proporcione los recursos necesarios",
                "responder_con": ["sí", "no"]
            }
    
    async def handle_user_response(self, capability: str, response: str):
        """Manejar respuesta del usuario a una solicitud"""
        try:
            if "sí" in response.lower() or "si" in response.lower():
                logger.log("INFO", f"Usuario aprobó capacidad: {capability}", modulo="planificacion")
                # Crear entorno interno para la capacidad
                if "wallet" in capability.lower():
                    self._setup_wallet_infrastructure()
                elif "api" in capability.lower():
                    self._setup_api_infrastructure()
                # Marcar como implementada
                ruleset.add_requested_capability(capability)
            else:
                logger.log("INFO", f"Usuario rechazó capacidad: {capability}", modulo="planificacion")
                # Posponer por 7 días
                self.request_cooldown[capability] = time.time() + 604800
            
            # Registrar decisión
            decision_id = str(uuid.uuid4())
            await self.memory.set_data(
                f"capability:{capability}:decision", 
                json.dumps({
                    "capability": capability,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
            )
        except Exception as e:
            logger.log("ERROR", f"Error manejando respuesta: {str(e)}", modulo="planificacion")
    
    def _setup_wallet_infrastructure(self):
        """Configurar entorno para manejo de wallets"""
        logger.log("INFO", "Configurando infraestructura para wallet", modulo="planificacion")
        # Lógica para crear estructura de directorios y archivos base
        os.makedirs("/storage/emulated/0/legion/wallets", exist_ok=True)
        with open("/storage/emulated/0/legion/wallets/README.txt", "w") as f:
            f.write("Directorio para almacenamiento seguro de wallets")
    
    def _setup_api_infrastructure(self):
        """Configurar entorno para integración de APIs"""
        logger.log("INFO", "Configurando infraestructura para APIs", modulo="planificacion")
        # Crear estructura para gestión de APIs
        os.makedirs("/storage/emulated/0/legion/apis", exist_ok=True)
        with open("/storage/emulated/0/legion/apis/config_template.json", "w") as f:
            f.write('{"api_name": "", "endpoint": "", "authentication": {}}')
    
    async def execute_planning_cycle(self):
        """Ciclo completo de planificación estratégica"""
        try:
            logger.log("INFO", "Iniciando ciclo de planificación estratégica", modulo="planificacion")
            
            # Detectar capacidades faltantes
            capabilities = await self.detect_structural_blockers()
            if not capabilities:
                logger.log("INFO", "No se detectaron limitaciones críticas", modulo="planificacion")
                return
            
            for capability in capabilities:
                # Saltar si ya fue solicitada recientemente
                if ruleset.is_capability_requested(capability):
                    continue
                
                # Evaluar si es crítica
                if await self.should_request_help(capability):
                    # Generar solicitud
                    request_data = await self.generate_request_for_user(capability)
                    
                    # Validar con IA triple
                    if not await self.core.is_action_legal(json.dumps(request_data)):
                        logger.log("WARNING", f"Solicitud para '{capability}' rechazada por validación IA", modulo="planificacion")
                        continue
                    
                    # Mostrar al usuario
                    await self.interface.display_request(request_data)
                    
                    # Registrar solicitud
                    ruleset.add_requested_capability(capability)
                    self.request_cooldown[capability] = time.time()
            
            logger.log("SUCCESS", "Ciclo de planificación completado", modulo="planificacion")
        except Exception as e:
            logger.log("ERROR", f"Error en ciclo de planificación: {str(e)}", modulo="planificacion")

# ===================== FUNCIONES DEL SISTEMA =====================
def get_system_stats() -> Tuple[float, float]:
    """Obtener estadísticas del sistema de forma segura"""
    try:
        # Obtener uso de memoria en Android
        recovery = SelfRecoveryProtocol(None, None, None)
        mem_stats = recovery.get_android_memory_stats()
        total = int(mem_stats.get('MemTotal', '0 kB').split()[0])
        free = int(mem_stats.get('MemFree', '0 kB').split()[0])
        ram_percent = ((total - free) / total) * 100 if total > 0 else 0
        
        # CPU no disponible en Android sin root
        cpu_percent = 0.0
        
        return ram_percent, cpu_percent
    except Exception:
        return 0.0, 0.0

async def simulate_corruption():
    """Simular corrupción para probar sistema de recuperación"""
    try:
        fs_manager = FileSystemManager(SecurityManager())
        if not await fs_manager.acquire_lock("simulate_corruption"):
            logger.log("WARNING", "No se pudo adquirir lock para simulación", modulo="test")
            return False
        
        with open(__file__, 'a') as f:
            f.write("\n# CORRUPTED_INJECTION")
        
        logger.log("CRITICAL", "Corrupción simulada exitosamente", modulo="test")
        return True
    except Exception as e:
        logger.log("ERROR", f"Error en simulación de corrupción: {str(e)}", modulo="test")
        return False
    finally:
        await fs_manager.release_lock("simulate_corruption")

# ===================== INICIALIZACIÓN DEL SISTEMA =====================
async def system_init():
    """Inicialización completa del sistema LEGIÓN OMEGA"""
    logger.log("INFO", "Iniciando sistema LEGIÓN OMEGA v9.4", modulo="core")
    
    security_manager = SecurityManager()
    memory = DistributedMemory()
    await memory.connect()
    fs_manager = FileSystemManager(security_manager)
    
    recovery_protocol = SelfRecoveryProtocol(memory, security_manager, fs_manager)
    
    model_engine = IAOrchestrator(security_manager)
    cognitive_core = CognitiveCore(security_manager, model_engine)
    
    decision_memory = DecisionMemory(memory)
    github_integration = GitHubIntegration(security_manager, fs_manager)
    user_interface = UserCommunicationInterface(security_manager, decision_memory)
    
    evolution_manager = EvolutionManager(
        security_manager, 
        fs_manager, 
        memory, 
        cognitive_core,
        decision_memory,
        github_integration
    )
    
    longterm_planner = LongTermPlanner(memory, cognitive_core, user_interface)
    
    logger.log("INFO", "Sistema completamente inicializado", modulo="core")
    
    return {
        "security_manager": security_manager,
        "memory": memory,
        "fs_manager": fs_manager,
        "recovery_protocol": recovery_protocol,
        "model_engine": model_engine,
        "cognitive_core": cognitive_core,
        "decision_memory": decision_memory,
        "github_integration": github_integration,
        "user_interface": user_interface,
        "evolution_manager": evolution_manager,
        "longterm_planner": longterm_planner
    }

# ===================== FUNCIÓN PRINCIPAL =====================
async def main():
    """Punto de entrada principal del sistema"""
    try:
        system = await system_init()
        logger.log("SUCCESS", "LEGIÓN OMEGA v9.4 operativo", modulo="core")
        
        # Iniciar verificación periódica de integridad
        asyncio.create_task(periodic_integrity_check(system["security_manager"], system["recovery_protocol"]))
        
        cycle_count = 0
        while True:
            # Ejecutar ciclo de evolución cada 6 horas
            if cycle_count % 6 == 0:
                await system["evolution_manager"].execute_evolution_cycle()
            
            # Ejecutar planificación estratégica cada 5 minutos
            if cycle_count % 1 == 0:
                await system["longterm_planner"].execute_planning_cycle()
            
            # Comunicación con el usuario
            await check_user_communication(system["user_interface"], system["cognitive_core"])
            
            # Manejar CLI en Termux
            if IS_MOBILE:
                await handle_termux_cli(system["user_interface"], system["cognitive_core"])
            
            # Verificar salud del sistema
            health_status = await system["recovery_protocol"].check_system_health()
            if health_status["status"] != "healthy":
                await system["recovery_protocol"].repair_system(health_status)
            
            cycle_count += 1
            await asyncio.sleep(300)  # Ejecutar cada 5 minutos
    
    except KeyboardInterrupt:
        logger.log("INFO", "Apagando sistema LEGIÓN OMEGA", modulo="core")
    except Exception as e:
        logger.log("CRITICAL", f"Error fatal: {str(e)}", modulo="core")
        traceback.print_exc()
        
        with open(FAIL_LOG_FILE, "a") as f:
            f.write(f"\n\n===== FATAL ERROR @ {datetime.now()} =====\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write(f"Code Hash: {SecurityManager().triple_hash(open(__file__,'rb').read())}\n")
            f.write("="*50 + "\n")
        
        if 'recovery_protocol' in locals():
            await system["recovery_protocol"].activate_recovery_protocol()

async def periodic_integrity_check(security_manager: SecurityManager, recovery_protocol: SelfRecoveryProtocol):
    """Verificación periódica de integridad del sistema"""
    while True:
        try:
            if not security_manager.verify_integrity():
                logger.log("CRITICAL", "¡Integridad comprometida detectada!", modulo="security")
                await recovery_protocol.activate_recovery_protocol()
            await asyncio.sleep(300)  # Verificar cada 5 minutos
        except Exception as e:
            logger.log("ERROR", f"Error en verificación de integridad: {str(e)}", modulo="security")
            await asyncio.sleep(300)

async def check_user_communication(interface: UserCommunicationInterface, cognitive_core: CognitiveCore):
    """Verificar si hay comunicación del usuario"""
    try:
        if os.path.exists("user_query.txt"):
            with open("user_query.txt", "r") as f:
                query = f.read()
            
            if query.strip():
                response, model_used = await interface.process_query(query, cognitive_core)
                with open("system_response.txt", "w") as f:
                    f.write(f"[{model_used.upper()}] {response}")
                
                os.remove("user_query.txt")
                
        # Verificar respuestas a solicitudes del planificador
        if os.path.exists("user_response.txt"):
            with open("user_response.txt", "r") as f:
                response = f.read().strip().lower()
            
            if response in ["sí", "si", "no"]:
                # Buscar la solicitud más reciente
                capability_keys = await cognitive_core.memory.get_all_keys("capability:*")
                if capability_keys:
                    latest_key = capability_keys[-1]
                    capability_data = json.loads(await cognitive_core.memory.get_data(latest_key))
                    capability = capability_data.get("capability", "")
                    
                    if capability:
                        await cognitive_core.longterm_planner.handle_user_response(capability, response)
                
                os.remove("user_response.txt")
    except Exception as e:
        logger.log("ERROR", f"Error procesando comunicación: {str(e)}", modulo="comunicacion")

async def handle_termux_cli(interface: UserCommunicationInterface, cognitive_core: CognitiveCore):
    """Manejar interfaz CLI para Termux con validación mejorada"""
    try:
        print("\nLEGIÓN OMEGA CLI - Escribe tu consulta (o 'test' para opciones de prueba):")
        query = input("> ")
        
        if query.lower() == 'exit':
            logger.log("INFO", "Saliendo de la CLI", modulo="comunicacion")
            return
        
        if query.lower() == 'test':
            print("\nOpciones de prueba:")
            print("1. Simular corrupción")
            print("2. Probar locks")
            print("3. Probar modo offline")
            print("4. Volver al menú principal")
            
            choice = input("Seleccione una opción: ")
            
            if choice == '1':
                if await simulate_corruption():
                    print("Corrupción simulada con éxito. El sistema debería recuperarse automáticamente.")
                else:
                    print("Error en simulación de corrupción")
            elif choice == '2':
                fs_manager = FileSystemManager(SecurityManager())
                if await fs_manager.acquire_lock("test_lock"):
                    print("Lock adquirido con éxito. Liberando en 5 segundos...")
                    await asyncio.sleep(5)
                    await fs_manager.release_lock("test_lock")
                    print("Lock liberado")
                else:
                    print("No se pudo adquirir el lock")
            elif choice == '3':
                global IS_OFFLINE
                IS_OFFLINE = True
                print("Modo offline activado. Las próximas operaciones no usarán Groq.")
            return
        
        response, model_used = await interface.process_query(query, cognitive_core)
        
        # Fragmentar respuesta si es necesario
        if len(response.split()) > MAX_TOKENS:
            parts = cognitive_core.split_message(response)
            for part in parts:
                print(f"\n{part}\n")
        else:
            print(f"\nRESPUESTA ({model_used.upper()}):\n{response}\n")
        
    except Exception as e:
        logger.log("ERROR", f"Error en CLI: {str(e)}", modulo="comunicacion")

if __name__ == "__main__":
    asyncio.run(main())
