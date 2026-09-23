from pathlib import Path
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image

ROOT = Path(r"C:\Drone")
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"
RES = ROOT / "android" / "app" / "src" / "main" / "res"
JAVA = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "drone" / "detector"
MODEL = ROOT / "android" / "app" / "src" / "main" / "assets" / "best_v5_int8.tflite"
if not MODEL.exists():
    MODEL = ROOT / "best_v5_int8.tflite"

LOGO = ROOT / "logo.png"


def bts(data: bytes) -> str:
    return ", ".join(str(b if b < 128 else b - 256) for b in data)


def encrypt_model() -> None:
    key = secrets.token_bytes(32)
    mask = secrets.token_bytes(32)
    stored = bytes(a ^ b for a, b in zip(key, mask))
    iv = secrets.token_bytes(12)
    plain = MODEL.read_bytes()
    packed = iv + AESGCM(key).encrypt(iv, plain, None)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "cfg.dat").write_bytes(packed)
    JAVA.joinpath("AssetPack.kt").write_text(
        """package com.drone.detector

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

internal object AssetPack {
    fun load(context: Context): ByteBuffer {
        val packed = context.assets.open(NAME).use { it.readBytes() }
        val iv = packed.copyOfRange(0, 12)
        val body = packed.copyOfRange(12, packed.size)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key(), "AES"), GCMParameterSpec(128, iv))
        val plain = cipher.doFinal(body)
        val buffer = ByteBuffer.allocateDirect(plain.size).order(ByteOrder.nativeOrder())
        buffer.put(plain)
        buffer.rewind()
        plain.fill(0)
        return buffer
    }

    private fun key(): ByteArray {
        val a = byteArrayOf(%s)
        val b = byteArrayOf(%s)
        return ByteArray(32) { i -> (a[i].toInt() xor b[i].toInt()).toByte() }
    }

    private const val NAME = "cfg.dat"
}
"""
        % (bts(stored), bts(mask)),
        encoding="utf-8",
    )
    print("encrypted", len(packed), "bytes")


def make_icons() -> None:
    src = Image.open(LOGO).convert("RGBA")
    drawable = RES / "drawable"
    drawable.mkdir(parents=True, exist_ok=True)
    src.resize((512, 512), Image.Resampling.LANCZOS).save(drawable / "ic_logo.png", "PNG")
    sizes = {
        "mipmap-mdpi": 48,
        "mipmap-hdpi": 72,
        "mipmap-xhdpi": 96,
        "mipmap-xxhdpi": 144,
        "mipmap-xxxhdpi": 192,
    }
    for folder, size in sizes.items():
        out = RES / folder
        out.mkdir(parents=True, exist_ok=True)
        img = src.resize((size, size), Image.Resampling.LANCZOS)
        img.save(out / "ic_launcher.png", "PNG")
        img.save(out / "ic_launcher_round.png", "PNG")
    print("icons written")


if __name__ == "__main__":
    encrypt_model()
    make_icons()
