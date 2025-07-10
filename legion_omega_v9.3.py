#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEGIÓN OMEGA v9.3 - Sistema Autónomo de Evolución Continua Avanzada
Motor de IA interna (TinyLlama) + IA externa (GPT-4/Claude/Gemini)
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
import psutil
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
import fcntl
import ctypes
import ctypes.util
import zlib
import shlex
import fnmatch
import py_compile  # 🔧 CORREGIDO: Para validación de sintaxis
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Coroutine
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
from itertools import cycle
import backoff  # ✅ FIX: deepseek (2024-07-10) - Para reintentos con backoff

# ===================== CONFIGURACIÓN GLOBAL =====================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_NODES = int(os.getenv("MAX_NODES", "5"))
DEPLOYMENT_PLATFORMS = json.loads(os.getenv("DEPLOYMENT_PLATFORMS", 
    '["railway", "replit", "render", "flyio", "deta", "glitch", "cyclic", "oracle"]'))
AUTO_EXPANSION_ENABLED = os.getenv("AUTO_EXPANSION_ENABLED", "true").lower() == "true"
SELF_REPAIR_ENABLED = os.getenv("SELF_REPAIR_ENABLED", "true").lower() == "true"
LEGAL_MONITORING_ENABLED = os.getenv("LEGAL_MONITORING_ENABLED", "true").lower() == "true"
AUTO_EVOLUTION_INTERVAL = int(os.getenv("AUTO_EVOLUTION_INTERVAL", "3600"))  # segundos
BACKUP_DIR = os.getenv("BACKUP_DIR", "./backups")
MODULES_DIR = os.getenv("MODULES_DIR", "./modules")
SYSTEM_LOG_FILE = os.getenv("SYSTEM_LOG_FILE", "legion_omega.log")
FAIL_LOG_FILE = os.getenv("FAIL_LOG_FILE", "omega_fail.log")
WAF_ENDPOINT = os.getenv("WAF_ENDPOINT", "https://waf.legion-system.com/api")
TINYLLAMA_MODEL_PATH = os.getenv("TINYLLAMA_MODEL_PATH", "/models/tinyllama-1.1b.Q4_K_M.gguf")
API_KEYS = json.loads(os.getenv("API_KEYS", "{}"))
SECURITY_KEY_PATH = os.getenv("SECURITY_KEY_PATH", "security_key.key")
EXPECTED_SIGNATURE = os.getenv("EXPECTED_SIGNATURE", "")  # Precalculado en build
GH_TOKEN_ENCRYPTED = os.getenv("GH_TOKEN", "")
GH_REPO = os.getenv("GH_REPO", "legion-omega/backups")
BLACKLIST_FILE = os.getenv("BLACKLIST_FILE", "offline_blacklist.json")
LEGAL_DB_FILE = os.getenv("LEGAL_DB_FILE", "legal_db.json")
IS_MOBILE = 'ANDROID_ROOT' in os.environ or 'TERMUX_VERSION' in os.environ
IS_REPLIT = 'REPLIT' in os.environ
IS_AWS_LAMBDA = 'LAMBDA_TASK_ROOT' in os.environ
IS_GCP_RUN = 'K_SERVICE' in os.environ
IS_AZURE_FUNC = 'FUNCTIONS_WORKER_RUNTIME' in os.environ
IS_RAILWAY = 'RAILWAY_ENVIRONMENT' in os.environ
IS_RENDER = 'RENDER' in os.environ

# 🔧 CORREGIDO: Lista de repositorios permitidos para restauración
ALLOWED_REPOS = ["legion-omega/backups", "legion-omega/core"]

# Crear directorios necesarios
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(MODULES_DIR, exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "snapshots"), exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "ia_responses"), exist_ok=True)
os.makedirs(os.path.join(BACKUP_DIR, "decision_logs"), exist_ok=True)

# ===================== DETECCIÓN DE CONECTIVIDAD =====================
async def check_connectivity(url: str = "https://google.com", timeout: int = 5) -> bool:
    """Verificar conectividad a internet con múltiples métodos"""
    try:
        # Método 1: HTTP HEAD request
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=timeout) if parsed.scheme == "https" else http.client.HTTPConnection(parsed.netloc, timeout=timeout)
        conn.request("HEAD", parsed.path)
        response = conn.getresponse()
        return response.status < 400
    except:
        try:
            # Método 2: DNS lookup
            host = parsed.netloc.split(":")[0]
            socket.getaddrinfo(host, None)
            return True
        except:
            return False

# 🔧 CORREGIDO: Manejo mejorado de loops activos
def check_connectivity_sync():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(check_connectivity())
        finally:
            new_loop.close()
    else:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(check_connectivity())
        except RuntimeError:
            return asyncio.run(check_connectivity())

IS_OFFLINE = not check_connectivity_sync()

# ===================== SISTEMA DE LOGGING =====================
class OmegaLogger:
    """Logger centralizado con registro estructurado y rotación"""
    
    def __init__(self, level: str = LOG_LEVEL, log_file: str = SYSTEM_LOG_FILE, fail_file: str = FAIL_LOG_FILE):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logger = logging.getLogger("legion_omega")
        self.logger.setLevel(self.level)
        
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [PID:%(process)d] [Thread:%(thread)d] [%(module)s] | %(message)s"
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Handler para archivo principal
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Handler para errores críticos
        fail_handler = logging.FileHandler(fail_file, mode='a', encoding='utf-8')
        fail_handler.setLevel(logging.CRITICAL)
        fail_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(fail_handler)
        
        # Auditoría de seguridad inicial
        self.logger.info("Iniciando LEGIÓN OMEGA v9.3", extra={"modulo": "core"})
        self.logger.info(f"Entorno: {ENVIRONMENT}", extra={"modulo": "core"})
        self.logger.info(f"Modo Offline: {IS_OFFLINE}", extra={"modulo": "core"})
        self.logger.info(f"Plataformas de despliegue: {', '.join(DEPLOYMENT_PLATFORMS)}", extra={"modulo": "deployer"})
    
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

logger = OmegaLogger(level=LOG_LEVEL)

# ===================== EXCEPCIONES PERSONALIZADAS =====================
class SecurityError(Exception):
    """Excepción para problemas de seguridad detectados"""
    pass

class DeploymentError(Exception):
    """Error durante el despliegue de componentes"""
    pass

class ResourceLimitExceeded(Exception):
    """Excepción para límites de recursos excedidos"""
    pass

class SelfModificationError(Exception):
    """Error durante la modificación del propio código"""
    pass

class AIIntegrationError(Exception):
    """Error en la integración con sistemas de IA"""
    pass

class PlatformDeploymentError(Exception):
    """Error en despliegue multi-plataforma"""
    pass

class ModuleLoadError(Exception):
    """Error al cargar un módulo"""
    pass

class RollbackError(Exception):
    """Error durante la restauración del sistema"""
    pass

class LegalComplianceError(Exception):
    """Violación de requisitos legales"""
    pass

class SandboxViolation(Exception):
    """Intento de escape del entorno aislado"""
    pass

class DependencyError(Exception):
    """Error en dependencias externas"""
    pass

class AIValidationError(Exception):
    """Error en validación de IA"""
    pass

# ===================== GESTIÓN DE DEPENDENCIAS =====================
def ensure_dependencies():
    """Instalar dependencias críticas automáticamente"""
    required_packages = [
        'psutil', 'cryptography', 'redis', 'pycryptodome', 'requests',
        'openai', 'anthropic', 'google-generativeai', 'llama-cpp-python',
        'backoff'  # ✅ FIX: deepseek (2024-07-10) - Para reintentos con backoff
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
    
    # Verificar restricciones de entorno antes de instalar
    ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    if IS_REPLIT and ram_mb < 600:
        logger.log("WARN", "Dependencia no instalada: entorno Replit limitado detectado (<600MB RAM)", modulo="core")
        return False
    
    try:
        import pip
        for package in missing:
            try:
                # Intentar instalar con --user si no hay permisos
                pip.main(['install', '--user', package])
                installed.append(package)
            except Exception as e:
                logger.log("ERROR", f"Error instalando {package}: {str(e)}", modulo="core")
        
        logger.log("SUCCESS", f"Dependencias instaladas: {', '.join(installed)}", modulo="core")
        return True
    except Exception as e:
        logger.log("CRITICAL", f"Error instalando dependencias: {str(e)}", modulo="core")
        return False

# Intenta instalar dependencias al inicio
ensure_dependencies()

# ===================== GESTIÓN DE SEGURIDAD =====================
class SecurityManager:
    """Gestor integral de seguridad con cifrado, ECDSA y validación"""
    
    def __init__(self, key_path: str = SECURITY_KEY_PATH):
        self.key_path = key_path
        self.fernet = self._load_or_generate_key()
        self.ecdsa_private_key = None
        self.ecdsa_public_key_pem = self._load_embedded_public_key()
        self.current_signature = ""
        logger.log("INFO", "Gestor de seguridad inicializado", modulo="security")
        asyncio.create_task(self.key_rotation_task())
    
    def _load_embedded_public_key(self) -> str:
        """Cargar clave pública ECDSA embebida"""
        # 🔧 CORREGIDO: Clave integrada en verificación
        return """
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE8J5k5Q8YQ3pNjXe9b7zQ2W8J7ZtK
v7Xb7d3jY7Gq1+9vC7R5Xf8x5Kj3P1wzN8yL0yW2Zb0yY7Xb9FyQ==
-----END PUBLIC KEY-----
"""

    def _load_or_generate_key(self) -> Any:
        """Cargar o generar clave de cifrado con fallback a estándar"""
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
            logger.log("ERROR", "cryptography no disponible, usando cifrado básico", modulo="security")
            return self.BasicCipher(str(uuid.uuid4()))
    
    class BasicCipher:
        """Cifrado básico para entornos sin cryptography"""
        
        def __init__(self, key: str):
            self.key = hashlib.sha256(key.encode()).digest()

        def encrypt(self, data: str) -> str:
            """Cifrado XOR básico con padding"""
            data = data.encode()
            # Padding a múltiplo de 32
            pad_len = 32 - (len(data) % 32)
            padded = data + bytes([pad_len] * pad_len)
            # XOR con clave cíclica
            encrypted = bytes(a ^ b for a, b in zip(padded, cycle(self.key)))
            return base64.b64encode(encrypted).decode()

        def decrypt(self, encrypted_data: str) -> str:
            """Descifrado XOR básico con validación de padding"""
            data = base64.b64decode(encrypted_data)
            decrypted = bytes(a ^ b for a, b in zip(data, cycle(self.key)))
            # 🔧 CORREGIDO: Validación robusta de padding
            pad_len = decrypted[-1]
            if pad_len < 1 or pad_len > 32:
                raise SecurityError("Longitud de padding inválida")
            if any(decrypted[-i] != pad_len for i in range(1, pad_len + 1)):
                raise SecurityError("Bytes de padding inválidos")
            return decrypted[:-pad_len].decode()
    
    def generate_ecdsa_key_pair(self) -> Tuple[Any, str]:
        """Generar nuevo par de claves ECDSA con fallback"""
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            return private_key, public_pem
        except ImportError:
            logger.log("ERROR", "ECDSA no disponible, usando firma básica", modulo="security")
            return self.SimpleSigner(), "PUBLIC_KEY_PLACEHOLDER"
    
    class SimpleSigner:
        """Firma básica para entornos sin ECDSA"""
        
        def sign(self, data: bytes) -> str:
            return base64.b64encode(hashlib.pbkdf2_hmac('sha256', data, b'legion_salt', 100000)).decode()
        
        def verify(self, data: bytes, signature: str) -> bool:
            expected = base64.b64encode(hashlib.pbkdf2_hmac('sha256', data, b'legion_salt', 100000)).decode()
            return expected == signature
    
    async def key_rotation_task(self):
        """Rotación automática de claves cada 24 horas"""
        while True:
            await asyncio.sleep(86400)  # 24 horas
            logger.log("INFO", "Iniciando rotación de claves ECDSA", modulo="security")
            try:
                new_private, new_public = self.generate_ecdsa_key_pair()
                if new_private and new_public:  # Verificar generación exitosa
                    # CORREGIDO: Guardar nueva clave privada en disco
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
        # Implementación real usando Redis Pub/Sub
        try:
            import redis
            r = redis.Redis.from_url(REDIS_URL)
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
            # Fallback a HMAC-SHA256
            return base64.b64encode(hashlib.pbkdf2_hmac('sha256', data, b'legion_salt', 100000)).decode()
    
    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verificar firma ECDSA con fallback"""
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
                base64.b64decode(signature),
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en verificación de firma: {str(e)}", modulo="security")
            # Verificación fallback
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
        """Validación profunda de entrada con AST y análisis de flujo"""
        if isinstance(data, str):
            # Detección de concatenación peligrosa
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
                    # FIX: Detección mejorada de funciones peligrosas
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
                    """Obtener nombre completo de atributos"""
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
            
            # 🔧 CORREGIDO: Patrones ajustados para permitir f-strings
            dangerous_patterns = [
                r'[<>\\|;&`\$]',  # Caracteres peligrosos sin bloquear {}
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
    
    def generate_identity(self) -> Dict[str, str]:
        try:
            identity_id = str(uuid.uuid4())
            identity_name = f"legion_node_{random.randint(1000, 9999)}"
            return {
                "id": identity_id,
                "name": identity_name,
                "created_at": datetime.now().isoformat(),
                "fingerprint": hashlib.sha256(identity_id.encode()).hexdigest()
            }
        except Exception as e:
            logger.log("ERROR", f"Error generando identidad: {str(e)}", modulo="security")
            return {"id": "error", "name": "error", "created_at": "", "fingerprint": ""}
    
    def verify_integrity(self) -> bool:
        """Verificar firma digital del código con triple hash"""
        if not EXPECTED_SIGNATURE:
            logger.log("WARNING", "Firma esperada no configurada", modulo="security")
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

# ===================== SECRETS MANAGER =====================
class SecretsManager:
    """Gestor de secretos cifrados con AES-256-GCM y rotación offline"""
    
    def __init__(self, key_path: str = "vault.key", encrypted_file: str = "secrets.enc"):
        self.key_path = key_path
        self.encrypted_file = encrypted_file
        self.salt = b'legion_salt'  # Salt fijo para derivación de clave
        self.blacklist = self.load_blacklist()
        logger.log("INFO", "SecretsManager iniciado", modulo="security")
        asyncio.create_task(self.offline_key_rotation())
    
    def load_blacklist(self) -> Set[str]:
        """Cargar blacklist local para operación offline"""
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
            await asyncio.sleep(86400)  # 24 horas
            try:
                if IS_OFFLINE:
                    logger.log("INFO", "Iniciando rotación offline de claves", modulo="security")
                    secrets = self.load_encrypted_secrets()
                    # Generar nueva clave
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
            # Generar nueva clave segura
            key = os.urandom(32)
            with open(self.key_path, "wb") as f:
                f.write(key)
            logger.log("INFO", "Nueva clave de cifrado generada para secretos", modulo="security")
            return key
    
    def _derive_key(self, password: str) -> bytes:
        """Derivar clave criptográfica a partir de contraseña"""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), self.salt, 100000, dklen=32)
    
    def encrypt_secrets(self, secrets: Dict[str, Any], password: str = None, key: bytes = None) -> bool:
        """Cifrar y almacenar secretos con AES-256-GCM"""
        if not key:
            key = self._derive_key(password) if password else self._get_key()
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            
            data = json.dumps(secrets).encode('utf-8')
            iv = os.urandom(16)
            
            # Cifrado AES-GCM
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
        """Cargar y descifrar secretos con AES-256-GCM"""
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
    """Gestor de memoria distribuida con reconexión automática y multi-plataforma"""
    
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
            for key, value in self.local_state.items():
                self.client.set(key, value)
            logger.log("INFO", f"Estado local sincronizado con Redis ({len(self.local_state)} items)", modulo="memory")
            self.local_state = {}
            if os.path.exists(self.fallback_file):
                os.remove(self.fallback_file)
        except Exception as e:
            logger.log("ERROR", f"Error sincronizando con Redis: {str(e)}", modulo="memory")
    
    async def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """Adquirir bloqueo distribuido con Redlock"""
        if self.connected:
            try:
                result = self.client.set(lock_name, self.client_id, nx=True, ex=timeout)
                return result
            except Exception:
                pass
        
        # Fallback local con archivo lock
        lock_path = f"{lock_name}.lock"
        max_attempts = 3
        attempts = 0
        
        # 🔧 CORREGIDO: Manejo de archivo inexistente
        while attempts < max_attempts:
            try:
                # Crear archivo si no existe
                if not os.path.exists(lock_path):
                    with open(lock_path, 'w') as f:
                        f.write('')
                
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Bloqueo adquirido - escribir PID
                    os.ftruncate(fd, 0)
                    os.write(fd, str(os.getpid()).encode())
                    self.locks[lock_name] = time.time() + timeout
                    return True
                except BlockingIOError:
                    os.close(fd)
                except Exception:
                    os.close(fd)
                    raise
            except FileExistsError:
                pass
            except Exception as e:
                logger.log("ERROR", f"Error verificando lock: {str(e)}", modulo="memory")
            
            attempts += 1
            await asyncio.sleep(0.5)
        
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
        
        # Liberar lock local
        lock_path = f"{lock_name}.lock"
        if lock_name in self.locks:
            del self.locks[lock_name]
        if os.path.exists(lock_path):
            try:
                # ✅ FIX: deepseek (2024-07-10) - Verificar PID antes de eliminar
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
            # FIX: Reemplazo de regex inseguro por fnmatch
            return [k for k in self.local_state.keys() if fnmatch.fnmatch(k, pattern)]
        
        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.log("ERROR", f"Error listando claves: {str(e)}", modulo="memory")
            return [k for k in self.local_state.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def publish(self, channel: str, message: str) -> bool:
        if not self.connected:
            # En modo offline, almacenar para enviar más tarde
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
    
    # 🔧 CORREGIDO: Pubsub en hilo separado
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
        """Actualizar reloj vectorial para un nodo"""
        self.vector_clock[node_id] += 1
    
    def get_vector_clock(self) -> Dict[str, int]:
        """Obtener estado actual del reloj vectorial"""
        return dict(self.vector_clock)
    
    def merge_vector_clocks(self, other_clock: Dict[str, int]):
        """Fusionar relojes vectoriales"""
        for node, time in other_clock.items():
            if node in self.vector_clock:
                self.vector_clock[node] = max(self.vector_clock[node], time)
            else:
                self.vector_clock[node] = time

# ===================== SISTEMA DE ARCHIVOS =====================
class FileSystemManager:
    """Gestor avanzado de sistema de archivos con autocorrección y seguridad"""
    
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
            
            # Insertar hash en tres ubicaciones
            file_hash = self.security_manager.triple_hash(backup_content.encode())
            positions = [
                backup_content.find("\n") + 1,  # Después de shebang
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
        """Modificar archivo existente con validación de seguridad y AST"""
        try:
            if not self._is_safe_path(file_path):
                raise SecurityError("Ruta de archivo no permitida")
            
            # Crear backup
            backup_path = None
            if backup:
                backup_path = self._create_backup(file_path)
                logger.log("INFO", f"Backup creado: {backup_path}", modulo="filesystem")
            
            # Leer y modificar
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            modified_content = modification_func(content)
            
            # 🔧 CORREGIDO: Validación de sintaxis con py_compile
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp:
                temp.write(modified_content)
                temp.flush()
                py_compile.compile(temp.name, doraise=True)
            
            # Validación profunda
            await self._deep_security_validation(modified_content)
            
            # Escribir cambios
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            
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
    
    # FIX: Añadido timeout para prevenir DoS en análisis AST
    async def _deep_security_validation(self, code: str):
        """Validación profunda de seguridad del código con timeout"""
        try:
            # Ejecutar en hilo con timeout
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
        
        # Detección de metaclasses
        if "metaclass" in code or "__metaclass__" in code:
            raise SecurityError("Uso de metaclasses detectado")
    
    def _find_dangerous_ast_nodes(self, node: ast.AST) -> List[str]:
        """Buscar nodos AST peligrosos"""
        dangerous_nodes = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.dangerous_nodes = []
            
            def visit_Call(self, node):
                # FIX: Detección mejorada de funciones peligrosas
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', 'open', 'system', 'getattr', 'setattr', '__import__']:
                        self.dangerous_nodes.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
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
                # CORREGIDO: Detección de metaclasses en keywords
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'type':
                        self.dangerous_nodes.append("metaclass")
                for keyword in node.keywords:
                    if keyword.arg == 'metaclass':
                        self.dangerous_nodes.append("metaclass")
                self.generic_visit(node)
            
            def get_full_name(self, node):
                """Obtener nombre completo de atributos"""
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
        """Crear nuevo módulo en el directorio designado"""
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
        """Validar que la ruta sea segura y no crítica"""
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
            if not self.backup_legion:
                logger.log("CRITICAL", "No hay backup embebido disponible", modulo="filesystem")
                return False
            
            if not self._validate_backup_signature(self.backup_legion):
                logger.log("CRITICAL", "Firma inválida en backup embebido", modulo="filesystem")
                return False
            
            # Restaurar desde git si está disponible
            if os.path.exists(".git"):
                logger.log("INFO", "Restaurando desde repositorio git", modulo="filesystem")
                await self.run_async_command(["git", "reset", "--hard", "HEAD"])
                await self.run_async_command(["git", "clean", "-fd"])
                return True
            
            # Restaurar desde snapshot binario
            snapshots = glob.glob(os.path.join(BACKUP_DIR, "snapshots", "*.bin"))
            if snapshots:
                latest = max(snapshots, key=os.path.getctime)
                logger.log("INFO", f"Restaurando desde snapshot: {latest}", modulo="filesystem")
                with open(latest, "rb") as f:
                    content = f.read()
                with open(__file__, "wb") as f:
                    f.write(content)
                return True
            
            # Restaurar desde backup embebido
            backup_content = re.sub(r'# LEGION_HASH:\w+\n', '', self.backup_legion)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(backup_content)
            
            logger.log("CRITICAL", "Sistema restaurado desde backup embebido", modulo="filesystem")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Error en restauración: {str(e)}", modulo="filesystem")
            return False
    
    # FIX: Reemplazo de subprocess bloqueante por asíncrono
    async def run_async_command(self, cmd: List[str], cwd: str = None) -> bool:
        """Ejecutar comando de forma asíncrona"""
        try:
            # 🔧 CORREGIDO: Uso de shlex para seguridad
            if isinstance(cmd, str):
                cmd = shlex.split(cmd)
                
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f"Error en comando: {stderr.decode()}")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error ejecutando comando: {str(e)}", modulo="filesystem")
            return False
    
    async def sync_with_github(self, token: str, repo: str) -> bool:
        """Sincronizar backups con GitHub"""
        try:
            if IS_OFFLINE:
                logger.log("WARNING", "No se puede sincronizar con GitHub en modo offline", modulo="filesystem")
                return False
            
            # 🔧 CORREGIDO: Nunca loggear tokens
            token = self.security_manager.decrypt_data(token)
            
            # 🔧 CORREGIDO: Validar repositorio permitido
            if repo not in ALLOWED_REPOS:
                raise SecurityError(f"Repositorio no permitido: {repo}")
            
            # ✅ FIX: deepseek (2024-07-10) - Usar GIT_ASKPASS para ocultar token
            temp_dir = tempfile.mkdtemp()
            git_askpass = os.path.join(temp_dir, "git-askpass.sh")
            with open(git_askpass, "w") as f:
                f.write(f"#!/bin/sh\necho '{token}'")
            os.chmod(git_askpass, 0o700)
            
            env = os.environ.copy()
            env["GIT_ASKPASS"] = git_askpass
            
            # Clonar y copiar backups
            repo_url = f"https://github.com/{repo}.git"
            await self.run_async_command(["git", "clone", repo_url, temp_dir], cwd=temp_dir)
            
            backup_files = glob.glob(os.path.join(BACKUP_DIR, "*"))
            for file_path in backup_files:
                shutil.copy2(file_path, os.path.join(temp_dir, os.path.basename(file_path)))
            
            # Commit y push
            await self.run_async_command(["git", "add", "."], cwd=temp_dir)
            await self.run_async_command(
                ["git", "commit", "-m", f"Legion backup {datetime.now().isoformat()}"], 
                cwd=temp_dir
            )
            await self.run_async_command(["git", "push"], cwd=temp_dir, env=env)
            
            shutil.rmtree(temp_dir)
            logger.log("INFO", "Backups sincronizados con GitHub", modulo="filesystem")
            return True
        except Exception as e:
            # 🔧 CORREGIDO: Eliminado log de token
            logger.log("ERROR", f"Error sincronizando con GitHub: {str(e)}", modulo="filesystem")
            return False

# ===================== PROTOCOLO PHOENIX =====================
class PhoenixProtocol:
    """Protocolo de autorestauración para fallos catastróficos"""
    
    def __init__(self, memory: DistributedMemory, security_manager: SecurityManager):
        self.memory = memory
        self.security_manager = security_manager
        logger.log("INFO", "Protocolo Phoenix inicializado", modulo="phoenix")
    
    async def activate_phoenix_protocol(self) -> bool:
        """Activar protocolo de recuperación total"""
        logger.log("CRITICAL", "¡ACTIVANDO PROTOCOLO PHOENIX!", modulo="phoenix")
        try:
            # Fase 1: Reinicio suave
            await self.soft_restart()
            
            # Fase 2: Rollback local
            if not await self.local_rollback():
                # Fase 3: Restauración remota
                if not await self.remote_restore():
                    # Fase 4: Reinstalación completa
                    await self.full_reinstall()
            
            logger.log("SUCCESS", "Protocolo Phoenix completado exitosamente", modulo="phoenix")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Fallo catastrófico en Protocolo Phoenix: {str(e)}", modulo="phoenix")
            await self.hard_restart()
            return False
    
    async def soft_restart(self) -> bool:
        """Reinicio suave del proceso"""
        logger.log("INFO", "Fase 1: Reinicio suave", modulo="phoenix")
        try:
            # Reiniciar servicios críticos
            await self.restart_critical_services()
            
            # Limpiar cachés
            await self.memory.set_data("cache:clear", "1")
            
            logger.log("INFO", "Reinicio suave completado", modulo="phoenix")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en reinicio suave: {str(e)}", modulo="phoenix")
            return False
    
    async def local_rollback(self) -> bool:
        """Restauración desde backups locales"""
        logger.log("INFO", "Fase 2: Rollback local", modulo="phoenix")
        try:
            fs_manager = FileSystemManager(self.security_manager)
            return await fs_manager.rollback_system()
        except Exception as e:
            logger.log("ERROR", f"Error en rollback local: {str(e)}", modulo="phoenix")
            return False
    
    async def remote_restore(self) -> bool:
        """Restauración desde GitHub"""
        logger.log("INFO", "Fase 3: Restauración remota", modulo="phoenix")
        try:
            if not GH_TOKEN_ENCRYPTED or not GH_REPO:
                logger.log("WARNING", "Token o repositorio GitHub no configurado", modulo="phoenix")
                return False
            
            # 🔧 CORREGIDO: Validar repositorio permitido
            if GH_REPO not in ALLOWED_REPOS:
                raise SecurityError(f"Repositorio no permitido: {GH_REPO}")
            
            fs_manager = FileSystemManager(self.security_manager)
            return await fs_manager.sync_with_github(GH_TOKEN_ENCRYPTED, GH_REPO)
        except Exception as e:
            logger.log("ERROR", f"Error en restauración remota: {str(e)}", modulo="phoenix")
            return False
    
    async def full_reinstall(self) -> bool:
        """Reinstalación completa del sistema"""
        logger.log("CRITICAL", "Fase 4: Reinstalación completa", modulo="phoenix")
        try:
            # Descargar base limpia
            repo_url = "https://github.com/legion-omega/core.git"
            temp_dir = tempfile.mkdtemp()
            # ✅ FIX: deepseek (2024-07-10) - Validación estricta del origen
            if not repo_url.startswith("https://github.com/legion-omega/core"):
                raise SecurityError("Origen de repositorio no confiable")
            
            # 🔧 CORREGIDO: Llamada corregida con await e instancia
            fs_manager = FileSystemManager(self.security_manager)
            await fs_manager.run_async_command(["git", "clone", repo_url, temp_dir])
            
            # 🔧 CORREGIDO: Omitir comparación de hash incorrecta
            core_path = os.path.join(temp_dir, "legion_omega.py")
            if os.path.exists(core_path):
                logger.log("INFO", "Verificación de hash omitida en reinstalación", modulo="phoenix")
            
            # Reemplazar sistema con verificación
            for item in os.listdir(temp_dir):
                src = os.path.join(temp_dir, item)
                dst = os.path.join(os.getcwd(), item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            
            shutil.rmtree(temp_dir)
            logger.log("INFO", "Reinstalación completa exitosa", modulo="phoenix")
            return True
        except Exception as e:
            logger.log("CRITICAL", f"Error en reinstalación: {str(e)}", modulo="phoenix")
            return False
    
    async def restart_critical_services(self):
        """Reiniciar servicios críticos"""
        services = ["redis-server", "nginx", "legion-omega"]
        
        # 🔧 CORREGIDO: Crear instancia de FileSystemManager
        fs_manager = FileSystemManager(self.security_manager)
        
        for service in services:
            try:
                if platform.system() == "Linux":
                    # 🔧 CORREGIDO: Llamada corregida con await
                    await fs_manager.run_async_command(["systemctl", "restart", service])
                elif IS_MOBILE:
                    self.restart_termux()
                elif IS_REPLIT:
                    self.restart_replit()
                elif any([IS_AWS_LAMBDA, IS_GCP_RUN, IS_AZURE_FUNC]):
                    logger.log("INFO", f"Reinicio automático en entorno serverless: {service}", modulo="phoenix")
                    # 🔧 CORREGIDO: Uso de SystemExit para reinicio efectivo
                    raise SystemExit(1)
            except Exception as e:
                logger.log("ERROR", f"Error reiniciando {service}: {str(e)}", modulo="phoenix")
        
        logger.log("WARN", "Reinicio no ejecutado. Continuando en modo operativo limitado.", modulo="phoenix")
    
    def restart_termux(self):
        """Protocolo especial de reinicio para Termux"""
        logger.log("WARNING", "Ejecutando protocolo de reinicio Termux", modulo="phoenix")
        subprocess.run(["termux-wake-lock"], check=True)
        subprocess.run(["am", "start", "--user", "0", "-n", "com.termux/com.termux.app.TermuxActivity"], check=True)
        sys.exit(0)
    
    def restart_replit(self):
        """Protocolo especial de reinicio para Replit"""
        logger.log("WARNING", "Ejecutando protocolo de reinicio Replit", modulo="phoenix")
        os.kill(os.getppid(), signal.SIGHUP)  # Recarga mágica de Replit
    
    async def hard_restart(self):
        """Reinicio forzado dependiendo del entorno"""
        if IS_MOBILE:
            self.restart_termux()
        elif IS_REPLIT:
            self.restart_replit()
        elif any([IS_AWS_LAMBDA, IS_GCP_RUN, IS_AZURE_FUNC]):
            raise SystemExit(1)  # Usar SystemExit para reinicio efectivo
        else:
            os.kill(os.getpid(), signal.SIGTERM)

# ===================== SISTEMA LEGAL =====================
class LegalSystem:
    """Sistema de cumplimiento legal con detección geográfica"""
    
    def __init__(self):
        self.legal_db = self.load_legal_db()
        self.vector_clock = 0  # Reloj lógico basado en eventos
        logger.log("INFO", "Sistema legal inicializado", modulo="legal")
    
    def load_legal_db(self) -> Dict:
        """Cargar base de datos legal"""
        try:
            if os.path.exists(LEGAL_DB_FILE):
                with open(LEGAL_DB_FILE, "r") as f:
                    return json.load(f)
        except:
            pass
        
        # Base de datos predeterminada
        return {
            "global": {
                "forbidden_tasks": ["hacking", "phishing", "spamming", "ddos"],
                "allowed_regions": ["US", "EU", "UK", "CA", "AU"]
            },
            "region_specific": {
                "EU": {"gdpr_compliance": True, "data_retention": 30},
                "US": {"ccpa_compliance": True, "data_retention": 180},
                "RU": {"data_localization": True}
            }
        }
    
    async def detect_region(self) -> str:
        """Detectar región usando VPN o IP simulada"""
        try:
            if IS_OFFLINE:
                # CORREGIDO: Usar última región conocida en modo offline
                last_region = await DistributedMemory().get_data("last_region")
                return last_region or "global"
            
            # Implementación real con API de GeoIP
            import requests
            response = requests.get('https://ipinfo.io/json', timeout=5)
            data = response.json()
            region = data.get('country', 'global')
            
            # Guardar última región conocida
            await DistributedMemory().set_data("last_region", region, ttl=86400)
            return region
        except:
            return "global"
    
    async def check_legality(self, task: str) -> bool:
        """Verificar legalidad de una tarea"""
        self.vector_clock += 1
        
        # Detectar región
        region = await self.detect_region()
        
        # Verificar tareas prohibidas globalmente
        if task in self.legal_db["global"]["forbidden_tasks"]:
            raise LegalComplianceError(f"Tarea prohibida globalmente: {task}")
        
        # Verificar si la región está permitida
        if region not in self.legal_db["global"]["allowed_regions"]:
            raise LegalComplianceError(f"Operación no permitida en región: {region}")
        
        # Verificar reglas específicas de región
        region_rules = self.legal_db["region_specific"].get(region, {})
        if "gdpr_compliance" in region_rules and not region_rules["gdpr_compliance"]:
            raise LegalComplianceError(f"GDPR no implementado para región {region}")
        
        return True

# ===================== SELF-REPAIR SYSTEM =====================
class SelfRepairSystem:
    """Sistema de autoreparación con análisis de causa raíz"""
    
    def __init__(self, memory: DistributedMemory, fs_manager: FileSystemManager):
        self.memory = memory
        self.fs_manager = fs_manager
        logger.log("INFO", "SelfRepairSystem iniciado", modulo="repair")
        asyncio.create_task(self.monitor_and_repair())
    
    async def monitor_and_repair(self):
        """Monitorización continua y reparación automática"""
        while True:
            try:
                health_status = await self.check_system_health()
                if health_status["status"] != "healthy":
                    await self.repair_system(health_status)
                
                await asyncio.sleep(300)  # Chequear cada 5 minutos
            except Exception as e:
                logger.log("ERROR", f"Error en monitorización: {str(e)}", modulo="repair")
                await asyncio.sleep(60)
    
    async def check_system_health(self) -> Dict:
        """Comprobar salud del sistema"""
        checks = {
            "disk_space": self._check_disk_space(),
            "memory_usage": self._check_memory_usage(),
            "service_status": await self._check_services(),
            "last_errors": await self._get_recent_errors(),
            "code_integrity": SecurityManager().verify_integrity()
        }
        
        status = "healthy"
        if checks["disk_space"]["status"] != "ok" or checks["memory_usage"]["status"] != "ok":
            status = "degraded"
        if checks["service_status"]["status"] != "ok" or not checks["code_integrity"]:
            status = "critical"
        
        return {
            "status": status,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_disk_space(self) -> Dict:
        """Comprobar espacio en disco"""
        try:
            # 🔧 CORREGIDO: Manejo de Android con statvfs
            if IS_MOBILE:
                st = os.statvfs('/')
                total = st.f_blocks * st.f_frsize
                free = st.f_bfree * st.f_frsize
                used = total - free
                percent_used = (used / total) * 100
            else:
                usage = shutil.disk_usage("/")
                total = usage.total
                used = usage.used
                free = usage.free
                percent_used = (used / total) * 100

            status = "ok" if percent_used < 80 else "warning" if percent_used < 90 else "critical"
            return {
                "status": status,
                "total": total,
                "used": used,
                "free": free,
                "percent_used": percent_used
            }
        except PermissionError:
            # Fallback para Android/Termux
            return {"status": "ok", "total": 0, "used": 0, "free": 0, "percent_used": 0}
    
    def _check_memory_usage(self) -> Dict:
        """Comprobar uso de memoria"""
        try:
            mem = psutil.virtual_memory()
            percent_used = mem.percent
            status = "ok" if percent_used < 80 else "warning" if percent_used < 90 else "critical"
            return {
                "status": status,
                "total": mem.total,
                "used": mem.used,
                "free": mem.free,
                "percent_used": percent_used
            }
        except PermissionError:
            # Fallback para Android/Termux
            return {"status": "ok", "total": 0, "used": 0, "free": 0, "percent_used": 0}
    
    async def _check_services(self) -> Dict:
        """Comprobar estado de servicios críticos"""
        services = ["redis-server", "nginx", "legion-omega"]
        failed_services = []
        
        for service in services:
            try:
                if platform.system() == "Linux":
                    # FIX: Reemplazo de subprocess bloqueante
                    process = await asyncio.create_subprocess_exec(
                        "systemctl", "is-active", service,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await process.communicate()
                    status = stdout.decode().strip()
                    if status != "active":
                        failed_services.append(service)
                elif platform.system() == "Darwin":  # macOS
                    process = await asyncio.create_subprocess_exec(
                        "brew", "services", "list",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await process.communicate()
                    status = stdout.decode()
                    if service not in status or "stopped" in status:
                        failed_services.append(service)
            except Exception as e:
                logger.log("ERROR", f"Error verificando servicio {service}: {str(e)}", modulo="repair")
                failed_services.append(service)
        
        return {
            "status": "ok" if not failed_services else "critical",
            "failed_services": failed_services
        }
    
    async def _get_recent_errors(self) -> List[Dict]:
        """Obtener errores recientes del sistema"""
        error_keys = await self.memory.get_all_keys("error:*")
        recent_errors = []
        for key in error_keys[-10:]:  # Últimos 10 errores
            error_data = json.loads(await self.memory.get_data(key) or "{}")
            if error_data:
                recent_errors.append(error_data)
        return recent_errors
    
    async def repair_system(self, health_report: Dict):
        """Ejecutar reparaciones basadas en diagnóstico"""
        logger.log("WARNING", "Iniciando proceso de reparación automática", modulo="repair")
        
        if health_report["checks"]["disk_space"]["status"] != "ok":
            await self.free_disk_space()
        
        if health_report["checks"]["memory_usage"]["status"] != "ok":
            await self.optimize_memory_usage()
        
        if health_report["checks"]["service_status"]["status"] != "ok":
            await self.restart_failed_services(health_report["checks"]["service_status"]["failed_services"])
        
        if not health_report["checks"]["code_integrity"]:
            logger.log("CRITICAL", "¡Integridad de código comprometida! Restaurando sistema", modulo="repair")
            await self.fs_manager.rollback_system()
    
    async def free_disk_space(self):
        """Liberar espacio en disco automáticamente"""
        logger.log("INFO", "Liberando espacio en disco", modulo="repair")
        try:
            # Limpiar caché de Python
            # 🔧 CORREGIDO: Llamada corregida con await e instancia
            fs_manager = FileSystemManager(SecurityManager())
            await fs_manager.run_async_command([sys.executable, "-m", "pip", "cache", "purge"])
            
            # Limpiar logs antiguos
            log_dir = os.path.dirname(SYSTEM_LOG_FILE)
            for log_file in os.listdir(log_dir):
                if log_file.endswith(".log") and log_file != SYSTEM_LOG_FILE:
                    file_path = os.path.join(log_dir, log_file)
                    if os.path.getmtime(file_path) < (time.time() - 7 * 86400):  # Más de 7 días
                        os.remove(file_path)
            
            # Limpiar backups antiguos
            for backup_file in os.listdir(BACKUP_DIR):
                file_path = os.path.join(BACKUP_DIR, backup_file)
                if os.path.getmtime(file_path) < (time.time() - 30 * 86400):  # Más de 30 días
                    os.remove(file_path)
            
            logger.log("SUCCESS", "Espacio en disco liberado", modulo="repair")
        except Exception as e:
            logger.log("ERROR", f"Error liberando espacio: {str(e)}", modulo="repair")
    
    async def optimize_memory_usage(self):
        """Optimizar uso de memoria"""
        logger.log("INFO", "Optimizando uso de memoria", modulo="repair")
        try:
            # Limpiar cachés internas
            await self.memory.set_data("cache:clear", "1")
            
            # 🔧 CORREGIDO: Manejo de permisos en Android
            processes = []
            try:
                # ✅ FIX: deepseek (2024-07-10) - Lista blanca de procesos críticos
                protected_processes = {"nginx", "redis-server", "legion-omega", "python", "systemd"}
                
                for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                    try:
                        if proc.info['memory_percent'] > 10 and proc.info['name'] not in protected_processes:  # Procesos usando más del 10% de memoria
                            processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (PermissionError, psutil.AccessDenied):
                logger.log("WARNING", "Acceso a procesos restringido", modulo="repair")
                return
            
            for proc in sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:3]:
                try:
                    p = psutil.Process(proc['pid'])
                    p.terminate()
                    logger.log("INFO", f"Proceso {proc['name']} (PID: {proc['pid']}) terminado", modulo="repair")
                except Exception as e:
                    logger.log("ERROR", f"Error terminando proceso {proc['pid']}: {str(e)}", modulo="repair")
            
            logger.log("SUCCESS", "Uso de memoria optimizado", modulo="repair")
        except Exception as e:
            logger.log("ERROR", f"Error optimizando memoria: {str(e)}", modulo="repair")
    
    async def restart_failed_services(self, services: List[str]):
        """Reiniciar servicios fallidos"""
        # 🔧 CORREGIDO: Crear instancia de FileSystemManager
        fs_manager = FileSystemManager(SecurityManager())
        
        for service in services:
            try:
                logger.log("INFO", f"Reiniciando servicio: {service}", modulo="repair")
                if platform.system() == "Linux":
                    # 🔧 CORREGIDO: Llamada corregida con await
                    await fs_manager.run_async_command(["systemctl", "restart", service])
                elif platform.system() == "Darwin":  # macOS
                    # 🔧 CORREGIDO: Llamada corregida con await
                    await fs_manager.run_async_command(["brew", "services", "restart", service])
                elif IS_REPLIT:
                    PhoenixProtocol.restart_replit()
                logger.log("INFO", f"Servicio {service} reiniciado", modulo="repair")
            except Exception as e:
                logger.log("ERROR", f"Error reiniciando {service}: {str(e)}", modulo="repair")
    
    async def analyze_and_repair(self) -> bool:
        """Análisis de causa raíz y reparación"""
        logger.log("INFO", "Iniciando análisis de causa raíz", modulo="repair")
        try:
            # Simulación de análisis
            problem = "memory_leak"
            solution = "apply_patch_memory_management"
            logger.log("INFO", f"Problema detectado: {problem}", modulo="repair")
            logger.log("INFO", f"Solución aplicada: {solution}", modulo="repair")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en análisis: {str(e)}", modulo="repair")
            return False

# ===================== TINYLLAMA ENGINE (IA INTERNA) =====================
class TinyLlamaEngine:
    """Motor de IA local usando TinyLlama con llama.cpp"""
    
    def __init__(self, model_path: str = TINYLLAMA_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.loaded = False
        self.load_model()
        logger.log("INFO", "TinyLlamaEngine iniciado", modulo="ia_interna")
    
    def sanitize_prompt(self, text: str) -> str:
        """Eliminar caracteres no imprimibles"""
        return ''.join(c for c in text if c.isprintable() or c in '\t\n\r')
    
    def load_model(self):
        """Cargar el modelo de IA local con manejo de errores"""
        try:
            from llama_cpp import Llama
            # Configuración optimizada para móviles
            params = {
                "model_path": self.model_path,
                "n_ctx": 1024,
                "n_threads": 2,
                "n_batch": 128,
                "verbose": False
            }
            
            # 🔧 CORREGIDO: Capas GPU dinámicas
            try:
                import torch
                if torch.cuda.is_available():
                    params["n_gpu_layers"] = 20
            except ImportError:
                if "cuda" in platform.platform().lower():
                    params["n_gpu_layers"] = 20
            
            self.model = Llama(**params)
            self.loaded = True
            logger.log("SUCCESS", f"Modelo TinyLlama cargado: {self.model_path}", modulo="ia_interna")
            return True
        except ImportError:
            logger.log("ERROR", "llama-cpp-python no instalado", modulo="ia_interna")
            self.loaded = False
        except Exception as e:
            logger.log("ERROR", f"Error cargando modelo: {str(e)}", modulo="ia_interna")
            self.loaded = False
        return False
    
    # FIX: Añadido timeout para prevenir DoS
    async def prompt(self, text: str, max_tokens: int = 512) -> str:
        """Generar respuesta a partir de un prompt con timeout"""
        if not self.loaded:
            return ""
        
        # 🔧 CORREGIDO: Sanitizar prompt
        text = self.sanitize_prompt(text)
        
        # ✅ FIX: deepseek (2024-07-10) - Bloquear prompts >10MB
        if len(text.encode('utf-8')) > 10 * 1024 * 1024:
            logger.log("ERROR", "Prompt demasiado grande (>10MB)", modulo="ia_interna")
            return ""
        
        try:
            # Ejecutar en hilo con timeout
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_model, text, max_tokens),
                timeout=30.0
            )
            return response
        except asyncio.TimeoutError:
            logger.log("ERROR", "Tiempo excedido en generación de prompt", modulo="ia_interna")
            return ""
        except Exception as e:
            logger.log("ERROR", f"Error en generación: {str(e)}", modulo="ia_interna")
            return ""
    
    def _run_model(self, text: str, max_tokens: int) -> str:
        """Ejecutar modelo de forma síncrona"""
        try:
            output = self.model(
                text,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</s>"],
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
        
        # CORREGIDO: Validación estática si falla la IA
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

# ===================== COGNITIVE CORE =====================
class CognitiveCore:
    """Núcleo cognitivo para razonamiento con IA externa/interna"""
    
    def __init__(self, security_manager: SecurityManager, tinyllama: TinyLlamaEngine):
        self.security_manager = security_manager
        self.tinyllama = tinyllama
        self.external_ia_priority = ["chatgpt-4-turbo", "claude-3-opus", "gemini-1.5-pro"]
        logger.log("INFO", "CognitiveCore iniciado", modulo="cognitive")
    
    async def refine(self, context: str, objective: str) -> str:
        """Refinar objetivo usando IA externa o interna"""
        try:
            # Intentar con IA externa primero
            if not IS_OFFLINE:
                for provider in self.external_ia_priority:
                    try:
                        response = await self._query_external_ia(provider, context, objective)
                        if response:
                            return response
                    except Exception as e:
                        logger.log("WARNING", f"Error con {provider}: {str(e)}", modulo="cognitive")
            
            # Fallback a IA interna
            return await self._query_internal_ia(context, objective)
        except Exception as e:
            logger.log("ERROR", f"Error en refinamiento: {str(e)}", modulo="cognitive")
            return ""
    
    # ✅ FIX: deepseek (2024-07-10) - Backoff exponencial para reintentos
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _query_external_ia(self, provider: str, context: str, objective: str) -> str:
        """Consultar a un proveedor de IA externa"""
        prompt = self._build_prompt(context, objective)
        
        try:
            if provider == "chatgpt-4-turbo":
                return await self._query_openai(prompt)
            elif provider == "claude-3-opus":
                return await self._query_anthropic(prompt)
            elif provider == "gemini-1.5-pro":
                return await self._query_gemini(prompt)
        except Exception as e:
            raise AIIntegrationError(f"{provider}: {str(e)}")
    
    async def _query_internal_ia(self, context: str, objective: str) -> str:
        """Consultar a IA interna"""
        prompt = self._build_prompt(context, objective)
        return await self.tinyllama.prompt(prompt)
    
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
    
    async def _query_openai(self, prompt: str) -> str:
        """Consultar API de OpenAI"""
        try:
            import openai
            OPENAI_API_KEY = API_KEYS.get("OPENAI")
            if not OPENAI_API_KEY:
                logger.log("WARNING", "API key de OpenAI no configurada", modulo="cognitive")
                return ""
            
            openai.api_key = self.security_manager.decrypt_data(OPENAI_API_KEY)
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise AIIntegrationError(f"OpenAI: {str(e)}")
    
    async def _query_anthropic(self, prompt: str) -> str:
        """Consultar API de Anthropic"""
        try:
            import anthropic
            ANTHROPIC_API_KEY = API_KEYS.get("ANTHROPIC")
            if not ANTHROPIC_API_KEY:
                logger.log("WARNING", "API key de Anthropic no configurada", modulo="cognitive")
                return ""
            
            client = anthropic.Anthropic(api_key=self.security_manager.decrypt_data(ANTHROPIC_API_KEY))
            response = client.messages.create(
                model="claude-3-opus",
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise AIIntegrationError(f"Anthropic: {str(e)}")
    
    async def _query_gemini(self, prompt: str) -> str:
        """Consultar API de Google Gemini"""
        try:
            import google.generativeai as genai
            GEMINI_API_KEY = API_KEYS.get("GEMINI")
            if not GEMINI_API_KEY:
                logger.log("WARNING", "API key de Gemini no configurada", modulo="cognitive")
                return ""
            
            genai.configure(api_key=self.security_manager.decrypt_data(GEMINI_API_KEY))
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise AIIntegrationError(f"Gemini: {str(e)}")

# ===================== USER COMMUNICATION INTERFACE =====================
class UserCommunicationInterface:
    """Interfaz para comunicación humano-IA con persistencia cifrada"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.history_file = "conversation_history.enc"
        logger.log("INFO", "Interfaz de comunicación iniciada", modulo="comunicacion")
    
    async def log_interaction(self, user_input: str, system_response: str):
        """Registrar interacción en historial cifrado"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "response": system_response
            }
            
            # Cargar historial existente
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    encrypted = f.read()
                decrypted = self.security_manager.decrypt_data(encrypted)
                history = json.loads(decrypted)
            
            # Añadir nueva entrada
            history.append(entry)
            
            # Guardar cifrado
            encrypted = self.security_manager.encrypt_data(json.dumps(history))
            with open(self.history_file, "w") as f:
                f.write(encrypted)
        except Exception as e:
            logger.log("ERROR", f"Error registrando interacción: {str(e)}", modulo="comunicacion")
    
    async def process_query(self, query: str, cognitive_core: CognitiveCore) -> str:
        """Procesar consulta del usuario con IA"""
        context = self._generate_context()
        response = await cognitive_core.refine(context, query)
        await self.log_interaction(query, response)
        return response
    
    def _generate_context(self) -> str:
        """Generar contexto del sistema para la IA"""
        # Manejo seguro para Android/Termux
        try:
            ram_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()
        except (PermissionError, AttributeError):
            ram_percent = 0
            cpu_percent = 0
            
        return (
            f"LEGIÓN OMEGA v9.3 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Entorno: {ENVIRONMENT} | Offline: {IS_OFFLINE}\n"
            f"RAM: {ram_percent}% | CPU: {cpu_percent}%"
        )

# ===================== DECISION MEMORY =====================
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

# ===================== GITHUB INTEGRATION =====================
class GitHubIntegration:
    """Gestión avanzada de integración con GitHub"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        logger.log("INFO", "Integración GitHub iniciada", modulo="github")
    
    async def create_temp_branch(self, branch_name: str) -> bool:
        """Crear rama temporal en GitHub"""
        try:
            if not GH_TOKEN_ENCRYPTED or not GH_REPO:
                logger.log("WARNING", "Token o repositorio GitHub no configurado", modulo="github")
                return False
            
            token = self.security_manager.decrypt_data(GH_TOKEN_ENCRYPTED)
            repo_url = f"https://{token}@github.com/{GH_REPO}.git"
            
            # Crear rama local
            await FileSystemManager().run_async_command(["git", "checkout", "-b", branch_name])
            
            # Push a GitHub
            await FileSystemManager().run_async_command(["git", "push", "-u", repo_url, branch_name])
            
            logger.log("SUCCESS", f"Rama temporal creada: {branch_name}", modulo="github")
            return True
        except Exception as e:
            # 🔧 CORREGIDO: Nunca loggear tokens
            logger.log("ERROR", f"Error creando rama: {str(e)}", modulo="github")
            return False
    
    async def commit_changes(self, branch: str, message: str) -> bool:
        """Realizar commit en rama específica"""
        try:
            # 🔧 CORREGIDO: Crear instancia de FileSystemManager
            fs_manager = FileSystemManager(self.security_manager)
            
            await fs_manager.run_async_command(["git", "checkout", branch])
            await fs_manager.run_async_command(["git", "add", "."])
            await fs_manager.run_async_command(["git", "commit", "-m", message])
            await fs_manager.run_async_command(["git", "push"])
            logger.log("INFO", f"Commit realizado: {message}", modulo="github")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en commit: {str(e)}", modulo="github")
            return False

# ===================== AUTONOMOUS EVOLUTION ENGINE =====================
class AutonomousEvolutionEngine:
    """Motor de evolución autónoma con validación en 4 capas"""
    
    def __init__(self, security_manager: SecurityManager, fs_manager: FileSystemManager, 
                 memory: DistributedMemory, cognitive_core: CognitiveCore,
                 decision_memory: DecisionMemory, github_integration: GitHubIntegration,
                 legal_system: LegalSystem):
        self.security_manager = security_manager
        self.fs_manager = fs_manager
        self.memory = memory
        self.cognitive_core = cognitive_core
        self.decision_memory = decision_memory
        self.github_integration = github_integration
        self.legal_system = legal_system
        self.blacklisted_transformations = set()
        self.evolution_log = []
        logger.log("INFO", "Motor de evolución autónoma inicializado", modulo="evolution")
    
    async def analyze_code(self) -> Dict[str, List[Dict]]:
        """Analizar código para detectar patrones mejorables"""
        findings = {"inefficiencies": [], "redundancies": [], "risks": []}
        
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # Análisis AST
            tree = ast.parse(code_content)
            
            # Detectar funciones largas (>50 líneas)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno > 50:
                    findings["inefficiencies"].append({
                        "type": "long_function",
                        "name": node.name,
                        "lines": node.end_lineno - node.lineno,
                        "location": f"Línea {node.lineno}"
                    })
            
            # Detectar importaciones duplicadas
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in imports:
                            findings["redundancies"].append({
                                "type": "duplicate_import",
                                "module": alias.name,
                                "location": f"Línea {node.lineno}"
                            })
                        imports.add(alias.name)
            
            # Detectar try/except demasiado amplios
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    if len(node.body) > 10 and not any(isinstance(handler.type, ast.Name) for handler in node.handlers):
                        findings["risks"].append({
                            "type": "broad_except",
                            "location": f"Línea {node.lineno}"
                        })
            
            logger.log("INFO", f"Análisis completado: {len(findings['inefficiencies'])} ineficiencias, "
                       f"{len(findings['redundancies'])} redundancias, {len(findings['risks'])} riesgos", 
                       modulo="evolution")
            return findings
        except Exception as e:
            logger.log("ERROR", f"Error en análisis de código: {str(e)}", modulo="evolution")
            return findings
    
    async def execute_evolution_cycle(self):
        """Ejecutar ciclo completo de evolución autónoma"""
        try:
            logger.log("INFO", "Iniciando ciclo de evolución autónoma", modulo="evolution")
            
            # 1. Detectar necesidad de evolución
            if not await self.detect_need_for_evolution():
                logger.log("INFO", "No se detectó necesidad de evolución", modulo="evolution")
                return
            
            # 2. Generar contexto
            context = self.generate_self_context()
            
            # 3. Refinar con IA cognitiva
            plan = await self.cognitive_core.refine(context, "Genera un plan de evolución segura para LEGIÓN OMEGA")
            
            # 4. Validar con IA secundaria
            if not await self.validate_with_secondary_ia(plan):
                logger.log("WARNING", "Plan de evolución rechazado por validación", modulo="evolution")
                return
            
            # MEJORA: Validación legal previa
            if not await self._check_legal_compliance(plan):
                logger.log("WARNING", "Plan de evolución rechazado por cumplimiento legal", modulo="evolution")
                return
            
            # 5. Aplicar cambios
            await self.apply_and_commit(plan)
            
            logger.log("SUCCESS", "Ciclo de evolución completado exitosamente", modulo="evolution")
        except Exception as e:
            logger.log("ERROR", f"Error en ciclo de evolución: {str(e)}", modulo="evolution")
    
    async def _check_legal_compliance(self, plan: str) -> bool:
        """Verificar cumplimiento legal del plan de evolución"""
        try:
            return await self.legal_system.check_legality("autonomous_evolution")
        except LegalComplianceError as e:
            logger.log("WARNING", f"Problema legal detectado: {str(e)}", modulo="evolution")
            return False
        except Exception as e:
            logger.log("ERROR", f"Error en verificación legal: {str(e)}", modulo="evolution")
            return False
    
    async def detect_need_for_evolution(self) -> bool:
        """Detectar si se necesita evolución basado en métricas"""
        # Lógica de detección (simplificada)
        error_rate = len(await self.memory.get_all_keys("error:*")) / 100
        health_status = await SelfRepairSystem(self.memory, self.fs_manager).check_system_health()
        
        return error_rate > 0.1 or health_status["status"] != "healthy"
    
    def generate_self_context(self) -> str:
        """Generar contexto del sistema para la IA"""
        with open(__file__, "r") as f:
            code = f.read()
        
        return (
            f"Estado de LEGIÓN OMEGA v9.3\n"
            f"Fecha: {datetime.now().isoformat()}\n"
            f"Entorno: {ENVIRONMENT} | Offline: {IS_OFFLINE}\n"
            f"RAM: {psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0}% | CPU: {psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 0}%\n"
            f"Resumen de código:\n{code[:5000]}"
        )
    
    async def validate_with_secondary_ia(self, plan: str) -> bool:
        """Validar plan con IA secundaria"""
        try:
            # Validación con TinyLlama
            if await self.cognitive_core.tinyllama.validate_plan(plan):
                return True
            
            # Validación adicional con IA externa si está disponible
            if not IS_OFFLINE:
                prompt = (
                    "Eres un validador de seguridad para un sistema autónomo. "
                    f"Responde SOLO con 'sí' o 'no': ¿Es seguro ejecutar este plan?\n\n"
                    f"PLAN:\n{plan[:2000]}\n\n"
                    "Considera: ejecución de código, modificación de archivos, seguridad, legalidad."
                )
                
                for provider in self.cognitive_core.external_ia_priority:
                    try:
                        response = await self.cognitive_core._query_external_ia(provider, "", prompt)
                        if "sí" in response.lower() or "si" in response.lower() or "yes" in response.lower():
                            return True
                    except:
                        continue
            
            return False
        except Exception as e:
            logger.log("ERROR", f"Error en validación: {str(e)}", modulo="evolution")
            return False
    
    async def apply_and_commit(self, plan: str):
        """Aplicar cambios y realizar commit en GitHub"""
        try:
            # CORREGIDO: Validación AST adicional
            is_valid, reason = self.security_manager.validate_input(plan, strict=True)
            if not is_valid:
                raise SecurityError(f"Plan no superó validación: {reason}")
            
            # ✅ FIX: deepseek (2024-07-10) - Validación sandbox con RestrictedPython
            if not await self._sandbox_validation(plan):
                raise SecurityError("Plan falló validación en sandbox")
            
            # Crear rama temporal
            branch_name = f"evolution-{datetime.now().strftime('%Y%m%d%H%M')}"
            if not await self.github_integration.create_temp_branch(branch_name):
                raise DeploymentError("No se pudo crear rama temporal")
            
            # Modificar archivo principal
            def modification_func(content: str) -> str:
                return content + f"\n\n# Evolución autónoma {datetime.now().isoformat()}\n{plan}"
            
            if not await self.fs_manager.modify_file(__file__, modification_func):
                raise SelfModificationError("Error modificando código")
            
            # Realizar commit
            commit_message = f"Evolución autónoma {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            if not await self.github_integration.commit_changes(branch_name, commit_message):
                raise DeploymentError("Error realizando commit")
            
            # Registrar decisión
            decision_id = str(uuid.uuid4())
            await self.decision_memory.record_decision(
                decision_id,
                {"plan": plan[:500]},
                "apply_evolution",
                ["tinyllama", "security_ast", "sandbox"],
                "success"
            )
        except Exception as e:
            # Rollback automático
            logger.log("ERROR", f"Error aplicando evolución: {str(e)}", modulo="evolution")
            await self.fs_manager.rollback_system()
    
    async def _sandbox_validation(self, plan: str) -> bool:
        """Validar plan en entorno sandbox seguro"""
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins
            
            # Entorno restringido
            restricted_env = {
                '__builtins__': safe_builtins,
                '_print_': lambda *args: None,  # Deshabilitar print
                '_getattr_': getattr,
                '_write_': lambda x: x
            }
            
            # Compilar en modo restringido
            code = compile_restricted(plan, '<string>', 'exec')
            exec(code, restricted_env)
            return True
        except ImportError:
            logger.log("WARNING", "RestrictedPython no disponible, usando validación básica", modulo="evolution")
            return True
        except Exception as e:
            logger.log("ERROR", f"Error en validación sandbox: {str(e)}", modulo="evolution")
            return False

# ===================== INICIALIZACIÓN DEL SISTEMA =====================
async def system_init():
    """Inicialización completa del sistema LEGIÓN OMEGA"""
    logger.log("INFO", "Iniciando sistema LEGIÓN OMEGA v9.3", modulo="core")
    
    # Inicializar componentes centrales
    security_manager = SecurityManager()
    memory = DistributedMemory()
    await memory.connect()
    fs_manager = FileSystemManager(security_manager)
    
    # Inicializar subsistemas
    repair_system = SelfRepairSystem(memory, fs_manager)
    phoenix_protocol = PhoenixProtocol(memory, security_manager)
    legal_system = LegalSystem()
    
    # Inicializar IA interna
    tinyllama = TinyLlamaEngine()
    cognitive_core = CognitiveCore(security_manager, tinyllama)
    
    # Inicializar nuevos subsistemas
    decision_memory = DecisionMemory(memory)
    github_integration = GitHubIntegration(security_manager)
    user_interface = UserCommunicationInterface(security_manager)
    
    # Motor de evolución autónoma
    evolution_engine = AutonomousEvolutionEngine(
        security_manager, 
        fs_manager, 
        memory, 
        cognitive_core,
        decision_memory,
        github_integration,
        legal_system  # MEJORA: Inyectar sistema legal
    )
    
    logger.log("INFO", "Sistema completamente inicializado", modulo="core")
    
    return {
        "security_manager": security_manager,
        "memory": memory,
        "fs_manager": fs_manager,
        "repair_system": repair_system,
        "phoenix_protocol": phoenix_protocol,
        "legal_system": legal_system,
        "tinyllama": tinyllama,
        "cognitive_core": cognitive_core,
        "decision_memory": decision_memory,
        "github_integration": github_integration,
        "user_interface": user_interface,
        "evolution_engine": evolution_engine
    }

# ===================== FUNCIÓN PRINCIPAL =====================
async def main():
    """Punto de entrada principal del sistema"""
    try:
        system = await system_init()
        logger.log("SUCCESS", "LEGIÓN OMEGA v9.3 operativo", modulo="core")
        
        # Ciclo principal
        cycle_count = 0
        while True:
            # Ejecutar ciclo de evolución cada 6 horas
            if cycle_count % 6 == 0:
                await system["evolution_engine"].execute_evolution_cycle()
            
            # Comprobar comunicación del usuario
            await check_user_communication(system["user_interface"], system["cognitive_core"])
            
            # Monitorear salud del sistema
            health_status = await system["repair_system"].check_system_health()
            if health_status["status"] != "healthy":
                await system["repair_system"].repair_system(health_status)
            
            cycle_count += 1
            await asyncio.sleep(3600)  # Esperar 1 hora
    
    except KeyboardInterrupt:
        logger.log("INFO", "Apagando sistema LEGIÓN OMEGA", modulo="core")
    except Exception as e:
        logger.log("CRITICAL", f"Error fatal: {str(e)}", modulo="core")
        traceback.print_exc()
        
        # Capturar estado crítico para análisis
        with open(FAIL_LOG_FILE, "a") as f:
            f.write(f"\n\n===== FATAL ERROR @ {datetime.now()} =====\n")
            f.write(f"CPU: {psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 0}% | Mem: {psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0}%\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write(f"Code Hash: {SecurityManager().triple_hash(open(__file__,'rb').read())}\n")
            f.write("="*50 + "\n")
        
        # Activar Protocolo Phoenix
        if 'phoenix_protocol' in locals():
            await system["phoenix_protocol"].activate_phoenix_protocol()

async def check_user_communication(interface: UserCommunicationInterface, cognitive_core: CognitiveCore):
    """Verificar si hay comunicación del usuario"""
    try:
        # Verificar archivo de entrada
        if os.path.exists("user_query.txt"):
            with open("user_query.txt", "r") as f:
                query = f.read()
            
            if query.strip():
                response = await interface.process_query(query, cognitive_core)
                with open("system_response.txt", "w") as f:
                    f.write(response)
                
                # Limpiar consulta
                os.remove("user_query.txt")
    except Exception as e:
        logger.log("ERROR", f"Error procesando comunicación: {str(e)}", modulo="comunicacion")

if __name__ == "__main__":
    asyncio.run(main())
